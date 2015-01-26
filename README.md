# Light Random Sprays Retinex Fork

This is a fork of the code provided at [http://www.fer.unizg.hr/ipg/resources/color_constancy](http://www.fer.unizg.hr/ipg/resources/color_constancy).

Use `make` to compile the code. **Note:** you have to adapt `main.cpp` in order to use the Light Random Sprays Retinex algorithm, currently only the Random Sprays Retinex algorithm is used.

Below you find the original README.

## Original README

Light Random Sprays Retinex is a simple image enhancing program for removing the local color cast of the scene illumination source and adjusting the brightness of a given image. The program itself represents a very good and fast approximation of the Random Sprays Retinex (RSR) algorithm providing resulting images in much less time and with more quality. It also allows the stronger effects of to be applied since it removes the RSR resulting noise. The program is based on the paper mentioned in the Literature section.

The program needs OpenCV library for image reading and simple processing. For compilation on Linux, the program also needs pkg-config.

### Version

This is version 0.1, the first version of Light Random Sprays Retinex. Bug reports and feedback are welcome.

### Download

You can always find the newest version at [http://www.fer.unizg.hr/ipg/resources/color_constancy/](http://www.fer.unizg.hr/ipg/resources/color_constancy/).

### License

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/).

### Literature

    [1] N. Banic and S. Loncaric.
        Light Random Sprays Retinex: Exploiting the Noisy Illumination Estimation.
        Signal Processing Letters, IEEE

Please send any comments, suggestions or bug fixes to Nikola Banic <nikola.banic@fer.hr>.

## License

See above for license information.

For license information of `Lenna.png` see [http://en.wikipedia.org/wiki/Lenna#mediaviewer/File:Lenna.png](http://en.wikipedia.org/wiki/Lenna#mediaviewer/File:Lenna.png).
