# Distributing Mimi outside the Mac App Store

Mimi releases must be signed with a **Developer ID Application** certificate,
submitted to Apple's notary service, and stapled before sharing them with other
people. Ad-hoc signatures are suitable only for local development.

## One-time Apple setup

1. Join the Apple Developer Program.
2. In Certificates, Identifiers & Profiles, create a **Developer ID
   Application** certificate and install it in Keychain Access.
3. Export the certificate and private key as a password-protected `.p12` file.
4. Create an App Store Connect API key with permission to use notarization and
   download its `.p8` private key once.

## GitHub secrets

Add these repository Actions secrets:

- `DEVELOPER_ID_APPLICATION_P12_BASE64`
- `DEVELOPER_ID_APPLICATION_P12_PASSWORD`
- `APP_STORE_CONNECT_API_KEY_P8_BASE64`
- `APP_STORE_CONNECT_API_KEY_ID`
- `APP_STORE_CONNECT_API_ISSUER_ID`

Encode binary credential files without line wrapping:

```sh
base64 -i DeveloperIDApplication.p12 | pbcopy
base64 -i AuthKey_XXXXXXXXXX.p8 | pbcopy
```

Pushing a `v*` tag then runs `.github/workflows/release.yml`. It imports the
certificate into an ephemeral keychain, builds a universal app with hardened
runtime and a secure timestamp, notarizes it with `notarytool`, staples the
ticket, verifies it with Gatekeeper, and replaces the GitHub release assets.

Never commit `.p12` or `.p8` files to the repository.
