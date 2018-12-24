# Computer Vision for Super Smash Bros Melee
This project runs a series of computer vision algorithms to analyze
the following famous [clip](https://www.youtube.com/watch?v=bj7IX18ccdY)
in which Moar Rzr Axe four stocks SilentWolf in less than at minute Evo 2014.

The analysis is fairly brittle and is not generalizable to other Super Smash Bros clips
The dmg percentage detection and in game timer detection still need some work and do
not work 100% of the time
## Dependencies
- Numpy
- opencv (using opencv-contrib)
- pytesseract
- tesseract-OCR

## Installation Instructions
### Installing Dependencies
`pip install numpy`

`pip install opencv-contrib-python`

`pip install pytesseract`

### Installing Tesseract-OCR

`pytesseract` is simply a wrapper for the CLI tool so you'll need to install
Tesseract-OCR for your platform as well. After you are done make sure Tesseract-OCR is added to your `PATH`.
You can confirm that everything works by running `tesseract --version`

For installation instructions see [here](https://github.com/tesseract-ocr/tesseract/wiki)

## Running the program

After you have all the dependencies installed you should be able
to run main.py. It will download the clip and start running the CV algorithms
to the console.