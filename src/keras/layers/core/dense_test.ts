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

  it('Dense layer: units = 1', () => {
    // let attrs = { 'name': 'fooLayer' };
    // let layer: Layer = new Layer(attrs);
    let dense_layer: Dense = new Dense(1);
    console.log('dense_layer =', dense_layer);

    let x1: Array2D = Array2D.new([2, 2], [[1, 2], [3, 4]]);
    let x2: Array2D = Array2D.new([3, 2], [[1, 2], [3, 4], [5, 6]]);
    let y: NDArray = dense_layer.call(x1);
    console.log("1. y =", y);  // DEBUG

    y = dense_layer.call(x2);
    console.log("2. y =", y);  // DEBUG
  });

});
