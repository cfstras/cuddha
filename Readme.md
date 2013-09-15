# cuddha

## Examples
_tbd._

## Overview
_cuddha_ is an attempt to build an extensive Buddhabrot renderer using the CUDA library for computation. I have built a [renderer in Java](https://github.com/cfstras/buddha) a while back, which was fun to play with but limited by the performance of Java and my CPU.
So, I decided to redo the whole thing in C++/CUDA from the ground up, learning something about CUDA and GPU performance considerations.

## Usage
**Warning:** If you are using the same GPU as CUDA and display device, this could cause your PC to hang or crash. I did my best to minimize this, but depending on the used parameters, the kernels could take a lot of GPU time. _Use at your own risk._

### Dependencies
Get these manually or by searching the package manager of your distro.

- CUDA 5.5
- libGLEW
- libGLFW3

If you are not running X server, you might need to tweak around with the linker settings in `Debug/objects.mk` to include your display server.

### Compiling and Running
I used Eclipse Nsight (bundled with the CUDA SDK) to build the project, which generates Makefiles in the _Debug_ directory.
If you don't have/want Nsight, the easiest way would be to have your CUDA installation in `/usr/local/cuda-5.5/` and then run `make` from the `Debug` folder:

```shell
ln -s /path/to/your/cuda/ /usr/local/cuda-5.5/ # only if different on your machine

git clone https://github.com/cfstras/cuddha
cd cuddha/Debug
make
./cuddha
```

When hitting the close button on the preview window, the current batch is finished (this could take up to 10 seconds) and a bitmap with the current state is saved to the output folder.

## Details
Some implementation details:
_tbd._

## Contributing
If you have ideas for different render modes or output conversions, I'm happy to receive mail!

If you want to code, just fork and pullrequest me once you have something interesting! :smile:

## License
The main code of this program is released under the MIT license, with the exception of the files `bmp.c` and `bmp.h`. These are released under the GNU GPLv3 License.
For details about those licenses, see the files `LICENSE_MIT.txt` and `LICENSE_GPL.txt`.
