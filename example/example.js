var ndarray = require('ndarray')
var ops = require('ndarray-ops')
var contact = require('../contact')
var getPixels = require('get-pixels')
var imshow = require('ndarray-imshow')

getPixels('puzzle.png', function(err,data) {
  var puzzle = data.pick(-1,-1,0)
  var outShape = [512,512]//[2*data.shape[0]+1, 2*data.shape[1]+1]
  var out = ndarray(
    new Float64Array(outShape[0]*outShape[1]),
    outShape)

  imshow(puzzle)

  contact(out, puzzle, puzzle, 1.0)
  imshow(out)

  contact(out, puzzle, puzzle, 16.0)
  imshow(out)

  contact(out, puzzle, puzzle, 32.0)
  imshow(out)  
})
