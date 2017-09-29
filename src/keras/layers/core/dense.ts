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

import { NDArrayMathCPU } from '../../../math/math_cpu';
import { NDArray, Array2D } from '../../../math/ndarray';
import { Layer } from '../../layer';


export class Dense extends Layer {

  protected units: number;
  protected activation: string;
  protected use_bias: boolean;
  protected kernel_initializer: string;
  protected bias_initializer: string;
  protected kernel_regularizer: string;
  protected bias_regularizer: string;
  protected kernel_constraint: string;
  protected bias_constraint: string;
  private input_last_dim: number;
  private kernel: Array2D;  // TODO(cais): Handle other ranks.
  private bias: Array2D;

  private math: NDArrayMathCPU;

  constructor(
    units: number,
    activation?: string,
    use_bias?: boolean,
    kernerl_initializer?: string,
    bias_initializer?: string,
    kernel_regularizer?: string,
    bias_regularizer?: string,
    activity_regularizer?: string,
    kernel_constraint?: string,
    bias_constraint?: string) {
    super();

    this.units = units;
    this.use_bias = use_bias
    if (this.use_bias == undefined) {
      this.use_bias = false;
    }

    console.log("this.kernel =", this.kernel);  // DEBUG
    console.log("this.bias =", this.bias);  // DEBUG
  }

  call(x: NDArray) {
    if (!(x instanceof Array2D)) {
      throw new Error(
        'Array2D is the only shape supported by Dense currently');
    }
    if (this.kernel == undefined) {
      // Lazy initialization of the kernel and bias.
      this.input_last_dim = x.shape[x.shape.length - 1];
      this.kernel = Array2D.zeros([this.input_last_dim, this.units]);
      // TODO(cais): Use this.kernel_initializer.

      if (this.use_bias) {
        this.bias = Array2D.zeros([1, this.units]);
      }

      this.math = new NDArrayMathCPU();
      // TODO(cais): Implement GPU math.
    } else {
      // TODO(cais): Check shape match?
    }

    return this.math.matMul(x, this.kernel);
  }

}
