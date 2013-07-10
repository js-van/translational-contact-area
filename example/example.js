var numeric = require("numeric")
var ndarray = require("ndarray")
var ops = require("ndarray-ops")
var contact = require("../contact.js")
var savePixels = require("save-pixels")
var fs = require("fs")
var morphology = require("ball-morphology")

function saveImg(filename, array) {
  var lo = ops.inf(array)
    , hi = ops.sup(array)
  console.log(lo, hi)
  ops.mulseq(ops.subseq(array, lo), 255.0/(hi-lo))
  savePixels(array, "png").pipe(fs.createWriteStream(filename))
}

require("get-pixels")("puzzle.png", function(err, data) {

  var puzzle = data.pick(-1, -1, 0)
  var nshape = numeric.sub(numeric.mul(puzzle.shape, 2), 1)
  var out = ndarray.zeros(nshape)
  
  saveImg("laplace.png", contact.laplace(out, puzzle, puzzle, 10.0))
  //saveImg("parallel.png", contact.parallel(out, puzzle, puzzle, 3.0))
  //saveImg("epsilon_boundary.png", contact.epsilonBoundary(out, puzzle, puzzle, 1.0))
})