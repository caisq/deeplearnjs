/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import { NDArrayMathGPU } from '../../../math/math_gpu';
import { Array2D, Array3D } from '../../../math/ndarray';
import { Flatten } from './flatten';

describe('Keras Flatten Layer Test', () => {
  let math: NDArrayMathGPU;
  beforeEach(() => {
    math = new NDArrayMathGPU();
    math.startScope();
  });

  afterEach(() => {
    math.endScope(null);
    math.dispose();
  });

  it('Flatten layer: Array2D input', () => {
    const flattenLayer: Flatten = new Flatten();

    const x1: Array2D = Array2D.new([2, 2], [[1, 2], [3, 4]]);
    const x2: Array2D = Array2D.new([3, 3], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    let y = flattenLayer.call(math, x1);
    expect(y.shape).toEqual([2, 2]);
    expect(y.getValues()).toEqual(new Float32Array([1, 2, 3, 4]));

    y = flattenLayer.call(math, x2);
    expect(y.shape).toEqual([3, 3]);
    expect(y.getValues()).toEqual(
      new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]));
  });

  it('Flatten layer: Array3D input', () => {
    const flattenLayer: Flatten = new Flatten();

    const x: Array3D = Array3D.new(
      [2, 2, 2], [[[-1, -2], [-3, -4]], [[1, 2], [3, 4]]]);
    const y = flattenLayer.call(math, x);
    expect(y.shape).toEqual([2, 4]);
    expect(y.getValues()).toEqual(
      new Float32Array([-1, -2, -3, -4, 1, 2, 3, 4]));
  });

});
