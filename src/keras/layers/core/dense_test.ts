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

import { Dense } from './dense';
import { Array2D, NDArray } from '../../../math/ndarray';

describe('Layer', () => {

  it('Dense layer: units=1', () => {
    let dense_layer: Dense = new Dense({ "units": 1 });

    let x1: Array2D = Array2D.new([2, 2], [[1, 2], [3, 4]]);
    // Changes in dimensions other than th last one should be okay.
    let x2: Array2D = Array2D.new([3, 2], [[1, 2], [3, 4], [5, 6]]);
    let x4: Array2D = Array2D.new([2, 3], [[1, 2, 3], [4, 5, 6]]);
    let y: NDArray = dense_layer.call(x1);
    expect(y.shape).toEqual([2, 1]);

    y = dense_layer.call(x2);
    expect(y.shape).toEqual([3, 1]);

    expect(() => dense_layer.call(x4)).toThrow();
  });

  it('Dense layer: units=1 activation=ReLU', () => {
    let dense_layer: Dense = new Dense({ "units": 1, "activation": "ReLU" });

    let x1: Array2D = Array2D.new([2, 2], [[1, 2], [3, 4]]);
    let x2: Array2D = Array2D.new([3, 2], [[1, 2], [3, 4], [5, 6]]);
    let x3: Array2D = Array2D.new([2, 3], [[1, 2, 3], [4, 5, 6]]);
    let y: NDArray = dense_layer.call(x1);
    expect(y.shape).toEqual([2, 1]);

    y = dense_layer.call(x2);
    expect(y.shape).toEqual([3, 1]);

    expect(() => dense_layer.call(x3)).toThrow();
  });

  it('Dense layer: units=1 no bias', () => {
    let dense_layer: Dense = new Dense({ "units": 1, "use_bias": false });

    let x1: Array2D = Array2D.new([2, 2], [[1, 2], [3, 4]]);
    let x2: Array2D = Array2D.new([3, 2], [[1, 2], [3, 4], [5, 6]]);
    let x3: Array2D = Array2D.new([2, 3], [[1, 2, 3], [4, 5, 6]]);
    let y: NDArray = dense_layer.call(x1);
    expect(y.shape).toEqual([2, 1]);

    y = dense_layer.call(x2);
    expect(y.shape).toEqual([3, 1]);

    expect(() => dense_layer.call(x3)).toThrow();
  });

  it('Dense layer: units=2', () => {
    let dense_layer: Dense = new Dense({ "units": 2 });

    let x1: Array2D = Array2D.new([2, 2], [[1, 2], [3, 4]]);
    let x2: Array2D = Array2D.new([3, 2], [[1, 2], [3, 4], [5, 6]]);
    let x3: Array2D = Array2D.new([2, 3], [[1, 2, 3], [4, 5, 6]]);
    let y: NDArray = dense_layer.call(x1);
    expect(y.shape).toEqual([2, 2]);

    y = dense_layer.call(x2);
    expect(y.shape).toEqual([3, 2]);

    expect(() => dense_layer.call(x3)).toThrow();
  });

});
