#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export PATH="$HOME/.cargo/bin:$PATH"

APP=Vox
BIN=vox
VER="${VER:-$(grep '^version' "$ROOT/Cargo.toml" | head -1 | sed 's/.*"\(.*\)"/\1/')}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"

ICON="$ROOT/assets/on.png"
[[ -f "$ICON" ]] || { echo "Icon not found: $ICON" >&2; exit 1; }

DIR="$CARGO_TARGET_DIR/dmg-build"
rm -rf "$DIR"
mkdir -p "$DIR"

ICON_SET_DIR="$DIR/icon.iconset"
mkdir -p "$ICON_SET_DIR"
for s in 16,16x16 32,16x16@2x 32,32x32 64,32x32@2x 128,128x128 256,128x128@2x 256,256x256 512,256x256@2x 512,512x512 1024,512x512@2x; do
    IFS=, read sz name <<< "$s"
    sips -z "$sz" "$sz" "$ICON" --out "$ICON_SET_DIR/icon_${name}.png" >/dev/null
done
iconutil -c icns "$ICON_SET_DIR" -o "$DIR/AppIcon.icns"
rm -rf "$ICON_SET_DIR"

ORT_VERSION=1.21.0
ORT_CACHE="$ROOT/.ort-cache"
fetch_ort() {
    local arch="$1"
    local ort_dir="$ORT_CACHE/$arch"
    [[ -f "$ort_dir/lib/libonnxruntime.dylib" ]] && return 0
    mkdir -p "$ort_dir"
    curl -fSL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-osx-${arch}-${ORT_VERSION}.tgz" | tar -xz -C "$ort_dir" --strip-components=1
}

fetch_ort arm64
fetch_ort x86_64

rustup target list --installed | grep -q "^aarch64-apple-darwin$" || rustup target add aarch64-apple-darwin
cargo build --release --target aarch64-apple-darwin

rustup target list --installed | grep -q "^x86_64-apple-darwin$" || rustup target add x86_64-apple-darwin
ORT_LIB_LOCATION="$ORT_CACHE/x86_64/lib" ORT_PREFER_DYNAMIC_LINK=1 cargo build --release --target x86_64-apple-darwin

package_dmg() {
    local target="$1" label="$2"
    local bundle="$DIR/$APP.app"
    local bin_path="$CARGO_TARGET_DIR/$target/release/$BIN"

    [[ -f "$bin_path" ]] || { echo "Binary not found: $bin_path" >&2; exit 1; }

    rm -rf "$bundle"
    mkdir -p "$bundle/Contents/MacOS" "$bundle/Contents/Resources"
    cp "$bin_path" "$bundle/Contents/MacOS/$BIN"
    cp "$DIR/AppIcon.icns" "$bundle/Contents/Resources/AppIcon.icns"

    # [[ -n "$dylib" ]] && {
    #     cp "$dylib" "$bundle/Contents/MacOS/"
    #     install_name_tool -change "@rpath/libonnxruntime.${ORT_VERSION}.dylib" \
    #         "@executable_path/libonnxruntime.${ORT_VERSION}.dylib" "$bundle/Contents/MacOS/$BIN"
    # }

    cat > "$bundle/Contents/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>$APP</string>
    <key>CFBundleDisplayName</key><string>$APP</string>
    <key>CFBundleIdentifier</key><string>com.vox.app</string>
    <key>CFBundleVersion</key><string>$VER</string>
    <key>CFBundleShortVersionString</key><string>$VER</string>
    <key>CFBundleExecutable</key><string>$BIN</string>
    <key>CFBundleIconFile</key><string>AppIcon</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
    <key>LSMinimumSystemVersion</key><string>11.0</string>
    <key>LSUIElement</key><true/>
    <key>NSHighResolutionCapable</key><true/>
    <key>NSMicrophoneUsageDescription</key><string>$APP requires microphone access for speech recognition.</string>
</dict>
</plist>
PLIST

    codesign --force --deep -s - "$bundle"

    local staging="$DIR/staging"
    rm -rf "$staging"
    mkdir -p "$staging"
    cp -R "$bundle" "$staging/"
    ln -s /Applications "$staging/Applications"
    hdiutil create -volname "$APP" -srcfolder "$staging" -ov -format ULMO "$DIR/$BIN-macos-$label.dmg"
    rm -rf "$staging" "$bundle"
}

package_dmg aarch64-apple-darwin arm64
package_dmg x86_64-apple-darwin x86_64

rm -f "$DIR/AppIcon.icns"
