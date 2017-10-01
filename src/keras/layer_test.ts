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

import { Layer } from './layer';
import { Array1D, NDArray } from '../math/ndarray';

describe('Layer', () => {

  it('Layer constructor and default call', () => {
    let layer: Layer = new Layer({});
    console.log('layer =', layer);

    let x: Array1D = Array1D.new([10, 20, 30]);
    let y: NDArray = layer.call(x);
    expect(y.getValues()).toEqual(new Float32Array([10, 20, 30]));
  });

});
