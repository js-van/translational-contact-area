"use strict"

var ndarray = require("ndarray")
var cwise = require("cwise")
var dup = require("dup")
var ops = require("ndarray-ops")
var fft = require("ndarray-fft")
var bits = require("bit-twiddle")
var correlate = require("ndarray-convolve").correlate
var morphology = require("ball-morphology")
var pool = require("typedarray-pool")


function pad(a, p) {
  var d = a.shape.length
    , nshape = new Array(d)
    , nstride = new Array(d)
    , nsize = 1
    , i
  for(i=d-1; i>=0; --i) {
    nshape[i] = a.shape[i] + 2 * p
    nstride[i] = nsize
    nsize *= nshape[i]
  }
  var t = pool.mallocDouble(nsize)
    , a_p = ndarray.ctor(t, nshape, nstride, 0)
  a_p.assigns(0)
  var a_l = a_p.lo.apply(a_p, dup([d], p))
  ops.assign(a_l.hi.apply(a_l, a.shape), a)
  return a_p
}

function do_correlate(out, a, b, p) {
  var d = a.shape.length
    , nshape = new Array(d)
    , nstride = new Array(d)
    , nsize = 1
    , i
  
  for(i=d-1; i>=0; --i) {
    nshape[i] = a.shape[i] + b.shape[i] - 2 * p
    nstride[i] = nsize
    nsize *= nshape[i]
  }
  
  var scratch_t = pool.mallocDouble(nsize)
    , scratch = ndarray.ctor(scratch_t, nshape, nstride, 0)
  
  correlate(scratch, a, b)
  
  scratch = scratch.lo.apply(scratch, dup([d], p))
  for(i=d-1; i>=0; --i) {
    scratch.shape[i] = Math.min(out.shape[i], scratch.shape[i] - p)
  }
  
  ops.assign(out.hi.apply(out, scratch.shape), scratch)
  pool.freeDouble(scratch_t)
}

function parallelContact(out, a, b, epsilon) {
  var p = Math.ceil(epsilon)|0
    , da = pad(a, p)
  morphology.dilate(da, epsilon)
  
  var db = pad(b, p)
  morphology.dilate(db, epsilon)

  do_correlate(out, da, db, p)
  
  pool.freeDouble(da.data)
  pool.freeDouble(db.data)
  
  return out
}

function epsilonBoundaryContact(out, a, b, epsilon) {

  var p = Math.ceil(epsilon)|0
    , da = pad(a, p)
    , ea = pad(a, p)
  morphology.dilate(da, epsilon)
  morphology.erode(ea, epsilon)
  ops.subeq(da, ea)
  pool.freeDouble(ea.data)
  
  var db = pad(b, p)
    , eb = pad(b, p)
  morphology.dilate(db, epsilon)
  morphology.erode(eb, epsilon)
  ops.subeq(db, eb)
  pool.freeDouble(eb.data)

  

  var as = ndarray.size(a)
    , ao = ndarray.order(a)
    , bs = ndarray.size(b)
    , bo = ndarray.order(b)

  var temp = pool.mallocDouble(Math.max(as, bs))

  var da_t = pool.mallocDouble(as)
    , da = ndarray(da_t, a.shape)
  
  ops.assign(da, a)
  morphology.dilate(da, epsilon)
  var ea = ndarray(temp, a.shape)
  ops.assign(ea, a)
  morphology.erode(ea, epsilon)
  ops.subeq(da, ea)
  
  
  var db_t = pool.mallocDouble(bs)
    , db = ndarray(db_t, b.shape, bo)
  ops.assign(db, b)
  morphology.dilate(db, epsilon)
  var eb = ndarray(temp, b.shape, bo)
  ops.assign(eb, b)
  morphology.erode(eb, epsilon)
  ops.subeq(db, eb)
  
  pool.freeDouble(temp)
  
  correlate(out, da, db)
  
  pool.freeDouble(da)
  pool.freeDouble(db)
  
  return out
}

var do_laplace = cwise({
  args: ["array", "array", "array", "array", "scalar", "index", "shape"],
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
    var w = -d2 / this.e2 * this.exp(-2.0 / this.e2 * d2)
    var a = +a_x
    var b = +a_y
    var c = +b_x
    var d = +b_y
    var k1 = c * (a + b)
    a_x = w*(k1 - b * (c - d))
    a_y = w*(k1 + a * (c + d))
  }
})

function laplaceContact(out, a, b, epsilon) {
  
  var pad = Math.ceil(epsilon)|0
    , d = a.shape.length
    , nshape = new Array(d)
    , nstride = new Array(d)
    , nsize = 1
    , i
  
  for(i=d-1; i>=0; --i) {
    nshape[i] = bits.nextPow2(a.shape[i] + b.shape[i] + 2 * pad)
    nstride[i] = nsize
    nsize *= nshape[i]
  }
  
  var a_x_t = pool.mallocDouble(nsize)
    , a_x = ndarray.ctor(a_x_t, nshape, nstride, 0)
  ops.assigns(a_x, 0)
  ops.assign(a_x.hi.apply(a_x, a.shape), a)
  
  var a_y_t = pool.mallocDouble(nsize)
    , a_y = ndarray.ctor(a_y_t, nshape, nstride, 0)
  ops.assigns(a_y, 0)
  
  fft(1, a_x, a_y)
  
  var b_x_t = pool.mallocDouble(nsize)
    , b_x = ndarray.ctor(b_x_t, nshape, nstride, 0)
  ops.assigns(b_x, 0)
  var b_x_l = b_x.lo.apply(b_x, a.shape)
  b_x_l = b_x_l.lo.apply(b_x_l, dup([d], pad))
  ops.assign(b_x_l.hi.apply(b_x_l, b.shape), b)
  
  var b_y_t = pool.mallocDouble(nsize)
    , b_y = ndarray.ctor(b_y_t, nshape, nstride, 0)
  ops.assigns(b_y, 0)
  
  fft(1, b_x, b_y)
  
  do_laplace(a_x, a_y, b_x, b_y, epsilon)
  
  pool.freeDouble(b_x_t)
  pool.freeDouble(b_y_t)
  
  fft(-1, a_x, a_y)
  
  var out_shape = new Array(d)
    , need_zero_fill = false
  for(i=0; i<d; ++i) {
    if(out_shape[i] > nshape[i] - 2*pad - 1) {
      need_zero_fill = true
      out_shape[i] = nshape[i] - 2*pad - 1
    } else {
      out_shape[i] = out.shape[i]
    }
  }
  if(need_zero_fill) {
    ops.assigns(out, 0)
  }
  a_x = a_x.lo.apply(a_x, dup([d], pad))
  ops.assign(out.hi.apply(out, out_shape), a_x.hi.apply(a_x, out_shape))
  
  pool.freeDouble(a_x_t)
  pool.freeDouble(a_y_t)
  
  return out
}

module.exports = laplaceContact
module.exports.laplace = laplaceContact
module.exports.parallel = parallelContact
module.exports.epsilonBoundary = epsilonBoundaryContact
