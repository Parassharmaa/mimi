#include "moonshine-c-api.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <sys/resource.h>

using Clock=std::chrono::steady_clock;
static double seconds(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
static std::string json(const std::string& s){std::string r="\"";for(unsigned char c:s){if(c=='"'||c=='\\'){r+='\\';r+=c;}else if(c=='\n')r+="\\n";else if(c=='\r')r+="\\r";else if(c=='\t')r+="\\t";else if(c<32)r+=' ';else r+=c;}return r+'"';}
static void check(int result){if(result<0)throw std::runtime_error(moonshine_error_to_string(result));}
static uint32_t u32(const unsigned char* p){uint32_t n;std::memcpy(&n,p,4);return n;}
static std::vector<float> wav(const char* path){
 std::ifstream f(path,std::ios::binary);if(!f)throw std::runtime_error("audio file missing");
 std::vector<unsigned char> b((std::istreambuf_iterator<char>(f)),{});
 if(b.size()<44||std::memcmp(b.data(),"RIFF",4)||std::memcmp(b.data()+8,"WAVE",4))throw std::runtime_error("expected RIFF WAV");
 int format=0,channels=0,rate=0,bits=0;size_t start=0,length=0;
 for(size_t o=12;o+8<=b.size();){uint32_t n=u32(b.data()+o+4);if(o+8+n>b.size())throw std::runtime_error("truncated WAV");
  if(!std::memcmp(b.data()+o,"fmt ",4)&&n>=16){format=b[o+8]|b[o+9]<<8;channels=b[o+10]|b[o+11]<<8;rate=u32(b.data()+o+12);bits=b[o+22]|b[o+23]<<8;}
  if(!std::memcmp(b.data()+o,"data",4)){start=o+8;length=n;}o+=8+n+(n%2);
 }
 if(channels!=1||rate!=16000||!length||!((format==1&&bits==16)||(format==3&&bits==32)))throw std::runtime_error("expected 16kHz mono PCM16 or float32");
 size_t width=bits/8;std::vector<float> samples(length/width);for(size_t i=0;i<samples.size();++i){if(format==3)std::memcpy(&samples[i],b.data()+start+i*width,width);else{int16_t value;std::memcpy(&value,b.data()+start+i*width,width);samples[i]=value/32768.0f;}}return samples;
}
static std::string text(transcript_t* result){std::string output;if(!result)return output;for(uint64_t i=0;i<result->line_count;++i){if(result->lines[i].text){output+=result->lines[i].text;}}return output;}
int main(int argc,char** argv){
 if(argc<5){std::cerr<<"usage: probe <model-dir> <arch> <wav> <direct|paced> [interval-seconds]\n";return 2;}
 int transcriber=-1,stream=-1;
 try{
  const bool paced=std::string(argv[4])=="paced";double interval=argc>5?std::stod(argv[5]):0.5;
  const char* steadyEnv=std::getenv("MIMI_MOONSHINE_STEADY_INTERVAL");double steady=steadyEnv?std::stod(steadyEnv):interval;
  auto prep=Clock::now();auto audio=wav(argv[3]);double preprocessing=seconds(prep);double duration=audio.size()/16000.0;
  std::string intervalString=std::to_string(interval);
  const char* provider=std::getenv("MIMI_MOONSHINE_PROVIDER");if(!provider)provider="CPU";
  const char* vadWindow=std::getenv("MIMI_MOONSHINE_VAD_WINDOW");if(!vadWindow)vadWindow="0.5";
  moonshine_option_t options[]={{"max_tokens_per_second","13.0"},{"transcription_interval",intervalString.c_str()},{"ort_providers",provider},{"vad_window_duration",vadWindow}};
  auto load=Clock::now();transcriber=moonshine_load_transcriber_from_files(argv[1],std::stoi(argv[2]),options,4,MOONSHINE_HEADER_VERSION);check(transcriber);double loading=seconds(load);
  // Kernel warm-up is separate and never contributes to the timed transcript.
  transcript_t* result=nullptr;auto warm=Clock::now();check(moonshine_transcribe_without_streaming(transcriber,audio.data(),std::min<size_t>(audio.size(),16000),16000,0,&result));double warming=seconds(warm);
  stream=moonshine_create_stream(transcriber,0);check(stream);check(moonshine_start_stream(transcriber,stream));
  auto start=Clock::now();double compute=preprocessing,first=-1,maxLateness=0;size_t next=static_cast<size_t>(interval*16000);std::string latest;
  std::vector<std::string> updates;std::vector<double> at;
  for(size_t o=0;o<audio.size();){size_t count=std::min<size_t>(1600,audio.size()-o);
   if(paced){auto deadline=start+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>((o+count)/16000.0));std::this_thread::sleep_until(deadline);maxLateness=std::max(maxLateness,std::chrono::duration<double>(Clock::now()-deadline).count());}
   auto work=Clock::now();check(moonshine_transcribe_add_audio_to_stream(transcriber,stream,audio.data()+o,count,16000,0));o+=count;
   if(o>=next){check(moonshine_transcribe_stream(transcriber,stream,0,&result));next=o+static_cast<size_t>(steady*16000);std::string output=text(result);if(!output.empty()&&output!=latest){latest=output;if(first<0)first=seconds(start);updates.push_back(output);at.push_back(seconds(start));}}
   compute+=seconds(work);
  }
  auto finish=Clock::now();check(moonshine_stop_stream(transcriber,stream));check(moonshine_transcribe_stream(transcriber,stream,0,&result));double finalization=seconds(finish);compute+=finalization;latest=text(result);double wall=seconds(start);
  if(first<0&&!latest.empty())first=wall;
  rusage usage{};getrusage(RUSAGE_SELF,&usage);
  std::cout<<"{\"mode\":"<<json(paced?"paced":"direct")<<",\"libraryVersion\":"<<moonshine_get_version()<<",\"audioDurationSeconds\":"<<duration<<",\"modelLoadSeconds\":"<<loading<<",\"warmupSeconds\":"<<warming<<",\"preprocessingSeconds\":"<<preprocessing<<",\"computeSeconds\":"<<compute<<",\"computeRTF\":"<<compute/duration<<",\"wallSeconds\":"<<wall<<",\"firstTextSeconds\":"<<(first<0?"null":std::to_string(first))<<",\"postInputFinalizationSeconds\":"<<finalization<<",\"maximumScheduleLatenessSeconds\":"<<maxLateness<<",\"peakRSSBytes\":"<<usage.ru_maxrss<<",\"finalText\":"<<json(latest)<<",\"updates\":[";
  for(size_t i=0;i<updates.size();++i){if(i)std::cout<<',';std::cout<<"{\"atSeconds\":"<<at[i]<<",\"text\":"<<json(updates[i])<<"}";}std::cout<<"]}\n";
  check(moonshine_free_stream(transcriber,stream));stream=-1;moonshine_free_transcriber(transcriber);return 0;
 }catch(const std::exception& e){if(stream>=0&&transcriber>=0)moonshine_free_stream(transcriber,stream);if(transcriber>=0)moonshine_free_transcriber(transcriber);std::cerr<<e.what()<<'\n';return 1;}
}
