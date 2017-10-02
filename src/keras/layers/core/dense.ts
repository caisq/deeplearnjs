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

import { NDArray, Array2D } from '../../../math/ndarray';
import { NDArrayMath } from '../../../math/math';
import { ActivationFunction, ReLUFunc, SigmoidFunc, TanHFunc } from
  '../../../math/activation_functions';
import { Initializer, VarianceScalingInitializer, ZerosInitializer } from
  '../../../initializers';
import { Layer } from '../../layer';

export class Dense extends Layer {

  protected units: number;
  protected activation: string;
  protected useBias: boolean;
  protected kernelInitializer: string;
  protected biasInitializer: string;
  protected kernelRegularizer: string;
  protected biasRegularizer: string;
  protected activityRegularizer: string;
  protected kernelConstraint: string;
  protected biasConstraint: string;

  private readonly DEFAULT_ACTIVATION: string = "linear";
  private readonly DEFAULT_USE_BIAS: boolean = true;
  private readonly DEFAULT_KERNEL_INITIALIZER = "glorot_uniform";
  private readonly DEFAULT_BIAS_INITIALIZER = "zeros";

  private inputLastDim: number;
  private kernel: Array2D;  // TODO(cais): Handle other ranks.
  private bias: Array2D;
  private kernelInitializerObject: Initializer;
  private biasInitializerObject: Initializer;
  private activationFunc: ActivationFunction;

  // tslint:disable-next-line:no-any
  constructor(attrs: any) {
    super(attrs);

    this.units = attrs.units;
    this.useBias = attrs.use_bias;
    if (this.useBias === undefined) {
      this.useBias = this.DEFAULT_USE_BIAS;
    }

    this.activation = (attrs.activation ||
      this.DEFAULT_ACTIVATION).toLowerCase();
    if (this.activation === "relu") {
      this.activationFunc = new ReLUFunc();
    } else if (this.activation === "sigmoid") {
      this.activationFunc = new SigmoidFunc();
    } else if (this.activation === "tanh") {
      this.activationFunc = new TanHFunc();
    } else if (this.activation !== this.DEFAULT_ACTIVATION) {
      throw new Error("Unsupported activation type: " + this.activation);
    }

    this.kernelInitializer = (attrs.kernel_initializer ||
      this.DEFAULT_KERNEL_INITIALIZER).toLowerCase();
    console.log("this.kernelInitializer =", this.kernelInitializer); // DEBUG
    if (this.kernelInitializer === "glorot_uniform") {
      this.kernelInitializerObject = new VarianceScalingInitializer(
        1.0, "fan_avg", "uniform");
    } else if (this.kernelInitializer === "glorot_normal") {
      this.kernelInitializerObject = new VarianceScalingInitializer(
        1.0, "fan_avg", "normal");
    } else if (this.kernelInitializer === "zeros") {
      this.kernelInitializerObject = new ZerosInitializer();
    } else {
      throw new Error(
        "Unsupporte kernel initializer: " + this.kernelInitializer);
    }

    this.biasInitializer = (attrs.bias_initializer ||
      this.DEFAULT_BIAS_INITIALIZER).toLowerCase();
    if (this.biasInitializer === "zeros") {
      this.biasInitializerObject = new ZerosInitializer();
    } else {
      throw new Error(
        "Unsupported bias initializer: " + this.biasInitializer);
    }

    this.kernelRegularizer = attrs.kernel_regularizer;
    this.biasRegularizer = attrs.bias_regularizer;
    this.activityRegularizer = attrs.activity_regularizer;
    this.kernelConstraint = attrs.kernel_constraint;
    this.biasConstraint = attrs.bias_constraint;

    // TODO(cais): Implement regularizers and constraints.
  }

  call(math: NDArrayMath, x: NDArray) {
    if (!(x instanceof Array2D)) {
      throw new Error(
        'Array2D is the only shape supported by Dense currently');
    }
    if (this.kernel === undefined) {
      // Lazy initialization of the kernel and bias.
      this.inputLastDim = x.shape[x.shape.length - 1];
      this.kernel = this.kernelInitializerObject.initialize(
        [this.inputLastDim, this.units],
        this.inputLastDim, this.units) as Array2D;

      if (this.useBias) {
        this.bias = Array2D.zeros([1, this.units]);
        this.bias = this.biasInitializerObject.initialize(
          [1, this.units], this.inputLastDim, this.units) as Array2D;
      }
    } else {
      // Check for shape mismatch.
      if (x.shape[x.shape.length - 1] !== this.inputLastDim) {
        throw new Error(
          'Last dimension of input (' + x.shape[x.shape.length - 1] +
          ') does not match first dimension of kernel (' +
          this.kernel.shape[0] + ').');
      }
    }

    let output: NDArray = math.matMul(x, this.kernel);
    if (this.useBias) {
      output = math.add(output, this.bias);
    }
    if (this.activationFunc !== undefined) {
      output = this.activationFunc.output(math, output);
    }
    return output;
  }

}
