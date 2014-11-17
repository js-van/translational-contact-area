'use strict'

module.exports = laplaceContact

var ndarray = require('ndarray')
var cwise   = require('cwise')
var dup     = require('dup')
var ops     = require('ndarray-ops')
var fft     = require('ndarray-fft')
var bits    = require('bit-twiddle')
var pool    = require('typedarray-pool')

var do_laplace = cwise({
  args: ['array', 'array', 'array', 'array', 'scalar', 'index', 'shape'],
  pre: function(a_x, a_y, b_x, b_y, epsilon, idx, shape) {
    this.d = shape.length
    this.e2 = epsilon * epsilon
    this.exp = Math.exp
  },
  body: function(a_x, a_y, b_x, b_y, epsilon, idx, shape) {
    var d2 = 0.0
    for(var i=0; i<this.d; ++i) {
      var s = shape[i]
      var f = idx[i]
      if(2*f > s) {
        f = s - f
      }
      var t = +f / +s
      d2 += t*t
    }

    var w = -d2 * this.exp(-2.0 * this.e2 * d2) / this.e2

    var a = +a_x
    var b = +a_y
    var c = +b_x
    var d = -b_y

    a_x = w*(a * c - b * d)
    a_y = w*(a * d + b * c)
  }
})

function zeros(nshape, nsize) {
  var t = pool.mallocDouble(nsize)
  for(var i=0; i<nsize; ++i) {
    t[i] = 0.0
  }
  return ndarray(t, nshape)
}

function image(f, nshape, nsize) {
  var r = zeros(nshape, nsize)
  ops.gts(r.hi.apply(r, f.shape.slice()), f, 0.0)
  return r
}

function fftImage(f, nshape, nsize) {
  var re = image(f, nshape, nsize)
  var im = zeros(nshape, nsize)
  fft(1, re, im)
  return [re, im]
}

function laplaceContact(out, a, b, epsilon) {
  var ashape  = a.shape.slice()
    , bshape  = b.shape.slice()
    , nshape  = out.shape.slice()
    , d       = a.shape.length
    , nsize   = 1
    , i
  
  for(i=d-1; i>=0; --i) {
    nsize *= nshape[i]
  }

  //Compute contact area
  var ah = fftImage(a, nshape, nsize)
  var bh = fftImage(b, nshape, nsize)
  do_laplace(ah[0], ah[1], bh[0], bh[1], epsilon)
  fft(-1, ah[0], ah[1])
  ops.assign(out, ah[0])

  //Release resources
  for(var i=0; i<2; ++i) {
    pool.free(ah[i].data)
    pool.free(bh[i].data)
  }
  
  return out
}