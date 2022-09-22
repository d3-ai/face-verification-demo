# Docker image
## Raspberry Pi4
Docker image for rapberry pi4 (Ubuntu20.04 LTS)

Edit a json Docker daemon configuration file
```=json
{
  "builder": {
    "gc": {
      "defaultKeepStorage": "20GB",
      "enabled": true
    }
  },
  "experimental": true, // change here if false
  "features": {
    "buildkit": true
  }
}
```
Build image for `linux/arm/v8`
```=bash
$ docker buildx build --platform linux/arm64/v8 --tag flower_mockup_pi4:latest --load ./baseimages/pi/
$ docker save -o {image_name}.tar {image_name}:latest
```
Install `{image_name}.tar` to Raspberry Pi
```=bash
$ docker load -i {image_name}.tar
```