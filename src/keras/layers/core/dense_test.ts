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
import { Dense } from './dense';
import { Array2D, NDArray } from '../../../math/ndarray';

describe('Dense Layer Test', () => {
  let math: NDArrayMathGPU;
  beforeEach(() => {
    math = new NDArrayMathGPU();
    math.startScope();
  });

  afterEach(() => {
    math.endScope(null);
    math.dispose();
  });

  it('Dense layer: units=1', () => {
    const denseLayer: Dense = new Dense(
      { "units": 1, "kernel_initializer": "ones", "bias_initializer": "zeros" }
    );

    const x1: Array2D = Array2D.new([2, 2], [[1, 2], [3, 4]]);
    // Changes in dimensions other than th last one should be okay.
    const x2: Array2D = Array2D.new([3, 2], [[1, 2], [3, 4], [5, 6]]);
    const x4: Array2D = Array2D.new([2, 3], [[1, 2, 3], [4, 5, 6]]);
    let y: NDArray = denseLayer.call(math, x1);
    expect(y.shape).toEqual([2, 1]);
    expect(y.getValues()).toEqual(new Float32Array([3, 7]));

    y = denseLayer.call(math, x2);
    expect(y.shape).toEqual([3, 1]);
    expect(y.getValues()).toEqual(new Float32Array([3, 7, 11]));

    expect(() => denseLayer.call(math, x4)).toThrow();
  });

  it('Dense layer: units=1, kernel_initializer=glorot_normal', () => {
    const denseLayer: Dense = new Dense(
      { "units": 1, "kernel_initializer": "glorot_normal" });

    const x1: Array2D = Array2D.new([2, 2], [[1, 2], [3, 4]]);
    // Changes in dimensions other than th last one should be okay.
    const x2: Array2D = Array2D.new([3, 2], [[1, 2], [3, 4], [5, 6]]);
    const x4: Array2D = Array2D.new([2, 3], [[1, 2, 3], [4, 5, 6]]);
    let y: NDArray = denseLayer.call(math, x1);
    expect(y.shape).toEqual([2, 1]);

    y = denseLayer.call(math, x2);
    expect(y.shape).toEqual([3, 1]);

    expect(() => denseLayer.call(math, x4)).toThrow();
  });

  it('Dense layer: units=1 activation=ReLU', () => {
    const denseLayer: Dense = new Dense({ "units": 1, "activation": "ReLU" });

    const x1: Array2D = Array2D.new([2, 2], [[1, 2], [3, 4]]);
    const x2: Array2D = Array2D.new([3, 2], [[1, 2], [3, 4], [5, 6]]);
    const x3: Array2D = Array2D.new([2, 3], [[1, 2, 3], [4, 5, 6]]);
    let y: NDArray = denseLayer.call(math, x1);
    expect(y.shape).toEqual([2, 1]);

    y = denseLayer.call(math, x2);
    expect(y.shape).toEqual([3, 1]);

    expect(() => denseLayer.call(math, x3)).toThrow();
  });

  it('Dense layer: units=1 no bias', () => {
    const denseLayer: Dense = new Dense({ "units": 1, "use_bias": false });

    const x1: Array2D = Array2D.new([2, 2], [[1, 2], [3, 4]]);
    const x2: Array2D = Array2D.new([3, 2], [[1, 2], [3, 4], [5, 6]]);
    const x3: Array2D = Array2D.new([2, 3], [[1, 2, 3], [4, 5, 6]]);
    let y: NDArray = denseLayer.call(math, x1);
    expect(y.shape).toEqual([2, 1]);

    y = denseLayer.call(math, x2);
    expect(y.shape).toEqual([3, 1]);

    expect(() => denseLayer.call(math, x3)).toThrow();
  });

  it('Dense layer: units=2', () => {
    const denseLayer: Dense = new Dense(
      { "units": 2, "kernel_initializer": "ones", "bias_initializer": "zeros" }
    );

    const x1: Array2D = Array2D.new([2, 2], [[1, 2], [3, 4]]);
    const x2: Array2D = Array2D.new([3, 2], [[1, 2], [3, 4], [5, 6]]);
    const x3: Array2D = Array2D.new([2, 3], [[1, 2, 3], [4, 5, 6]]);
    let y: NDArray = denseLayer.call(math, x1);
    expect(y.shape).toEqual([2, 2]);
    expect(y.getValues()).toEqual(new Float32Array([3, 3, 7, 7]));

    y = denseLayer.call(math, x2);
    expect(y.shape).toEqual([3, 2]);
    expect(y.getValues()).toEqual(new Float32Array([3, 3, 7, 7, 11, 11]));

    expect(() => denseLayer.call(math, x3)).toThrow();
  });

});
