translational-contact-area
===========================
This module computes the effective contact area between a pair of voxelized solids at some tolerance at a set of translational configurations.  The input solids are encoded as binary bitmaps (or voxel models) where <=0 means empty and >0 means filled.  The result is a field where each entry corresponds to the amount of effective surface contact between the two solids at the given relative translation.

# Example

```javascript
```

# API

#### `require('translational-contact-area')(out,a,b,epsilon)`


# Credit
(c) 2014 Mikola Lysenko. MIT License