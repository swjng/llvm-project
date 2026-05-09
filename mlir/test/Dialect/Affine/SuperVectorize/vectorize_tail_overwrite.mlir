// RUN: mlir-opt %s -affine-super-vectorize="virtual-vector-size=4" | FileCheck %s

// A non-reduction loop whose trip count is not a multiple of the vector
// factor: the trailing iteration's full-width `transfer_write` would
// otherwise clobber in-bounds memref cells outside the source loop's
// iteration range. The pass must mask the transfer ops using
// `vector.create_mask` so OOB lanes are not written.

// CHECK-LABEL: func @tail_overwrite_in_bounds
// CHECK:         affine.for %{{.*}} = 0 to 5 step 4 {
// CHECK:           %[[M:.*]] = vector.create_mask {{.*}} : vector<4xi1>
// CHECK:           vector.transfer_read {{.*}}, %[[M]] : memref<8xi32>, vector<4xi32>
// CHECK:           vector.transfer_write {{.*}}, %[[M]] : vector<4xi32>, memref<8xi32>
func.func @tail_overwrite_in_bounds(%a: memref<8xi32>) {
  affine.for %i = 0 to 5 {
    %v = affine.load %a[%i] : memref<8xi32>
    %d = arith.addi %v, %v : i32
    affine.store %d, %a[%i] : memref<8xi32>
  }
  return
}

// -----

// Trip count is a multiple of the vector factor: no mask needed.

// CHECK-LABEL: func @aligned_full_coverage
// CHECK:         affine.for %{{.*}} = 0 to 8 step 4 {
// CHECK-NOT:       vector.create_mask
// CHECK:           vector.transfer_read
// CHECK:           vector.transfer_write
func.func @aligned_full_coverage(%a: memref<8xi32>) {
  affine.for %i = 0 to 8 {
    %v = affine.load %a[%i] : memref<8xi32>
    %d = arith.addi %v, %v : i32
    affine.store %d, %a[%i] : memref<8xi32>
  }
  return
}
