# Fractal-Image-Compression

Simple implementation of a fractal-based compression algorithm introduced in *Solution of an inverse problem for fractals and other sets* by M. F. Barnsley & al. 

This implementation uses the Jacquard tiling of the image, in order to improve performances and/or reduce compute time, one should consider implementing tree representations of images. The algorithm uses IFS and the Collage Theorem to create an estimate of a given image. These estimates are stored within the linear transformations generated. 

### Example :

Here, the image shows the decompression process :

![alt text](https://github.com/FredericWantiez/Fractal-Image-Compression/blob/master/Report/images/monkey.png "Step of decompression")

The code implements both compression and decompression of a given grayscale image.
