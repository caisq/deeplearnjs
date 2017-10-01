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
import { ActivationFunction, ReLUFunc, SigmoidFunc, TanHFunc } from '../../../math/activation_functions';
import { Layer } from '../../layer';

export class Dense extends Layer {

  protected units: number;
  protected activation: string;
  protected use_bias: boolean;
  protected kernel_initializer: string;
  protected bias_initializer: string;
  protected kernel_regularizer: string;
  protected bias_regularizer: string;
  protected activity_regularizer: string;
  protected kernel_constraint: string;
  protected bias_constraint: string;

  private readonly default_activation: string = "linear";
  private readonly default_use_bias: boolean = true;

  private input_last_dim: number;
  private kernel: Array2D;  // TODO(cais): Handle other ranks.
  private bias: Array2D;
  private activation_func: ActivationFunction;

  private math: NDArrayMathCPU;

  constructor(attrs: any) {
    // units: number,
    // activation?: string,
    // use_bias?: boolean,
    // kernerl_initializer?: string,
    // bias_initializer?: string,
    // kernel_regularizer?: string,
    // bias_regularizer?: string,
    // activity_regularizer?: string,
    // kernel_constraint?: string,
    // bias_constraint?: string) {
    super(attrs);

    // TODO(cais): GPU case.
    this.math = new NDArrayMathCPU();

    this.units = attrs.units;
    this.use_bias = attrs.use_bias
    if (this.use_bias == undefined) {
      this.use_bias = this.default_use_bias;
    }
    this.activation = attrs.activation;
    if (this.activation == undefined) {
      this.activation = this.default_activation;
    }
    if (this.activation.toLowerCase() == "relu") {
      this.activation_func = new ReLUFunc();
    } else if (this.activation.toLowerCase() == "sigmoid") {
      this.activation_func = new SigmoidFunc;
    } else if (this.activation.toLowerCase() == "tanh") {
      this.activation_func = new TanHFunc();
    } else if (this.activation != this.default_activation) {
      throw new Error("Unsupported activation type: " + this.activation);
    }
    this.kernel_initializer = attrs.kernel_initializer;
    this.bias_initializer = attrs.bias_initializer;
    this.kernel_regularizer = attrs.kernel_regularizer;
    this.bias_regularizer = attrs.bias_regularizer;
    this.activity_regularizer = attrs.activity_regularizer;
    this.kernel_constraint = attrs.kernel_constraint;
    this.bias_constraint = attrs.bias_constraint;

    // TODO(cais): Implement regularizers and constraints.
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
    } else {
      // Check for shape mismatch.
      if (x.shape[x.shape.length - 1] != this.input_last_dim) {
        throw new Error(
          'Last dimension of input (' + x.shape[x.shape.length - 1] +
          ') does not match first dimension of kernel (' +
          this.kernel.shape[0] + ').');
      }
    }

    let output: NDArray = this.math.matMul(x, this.kernel);
    if (this.use_bias) {
      output = this.math.add(output, this.bias);
    }
    if (this.activation_func != undefined) {
      output = this.activation_func.output(this.math, output);
    }
    return output;
  }

}
