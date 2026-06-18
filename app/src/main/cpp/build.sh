set -e
cmake --preset android-release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build --preset android-release

mkdir -p lib
cp -r ./build/android/qnnlibs ../assets/
if [ -d ./build/android/mtklibs ]; then
  mkdir -p ../assets/mtklibs
  cp -r ./build/android/mtklibs/* ../assets/mtklibs/
  echo "MediaTek LiteRT libs copied to assets/mtklibs/"
else
  mkdir -p ../assets/mtklibs
  echo "Note: mtklibs not built (set LITERT_SDK_ROOT in CMake to bundle MediaTek libs)"
fi
mkdir -p ../jniLibs/arm64-v8a/
cp ./build/android/bin/arm64-v8a/libstable_diffusion_core.so ../jniLibs/arm64-v8a/
