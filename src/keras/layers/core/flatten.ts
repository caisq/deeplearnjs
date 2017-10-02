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

import { Layer } from '../../layer';
import { NDArray } from '../../../math/ndarray';
import { NDArrayMath } from '../../../math/math';

export class Flatten extends Layer {
  constructor() {
    super({});
  }

  // TOOD(cais): Move math object to keras.backend.
  call(math: NDArrayMath, x: NDArray): NDArray {
    let numelExcludingFirst = 1;
    // First dimension (batch) is excluded during Flatten calls.
    for (let i = 1; i < x.shape.length; ++i) {
      numelExcludingFirst *= x.shape[i];
    }
    return x.reshape([x.shape[1], numelExcludingFirst]);
  }
}
