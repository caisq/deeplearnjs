/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import {ENV} from '../environment';
import {keep, tidy, dispose} from '../globals';
import {scalar, zerosLike} from '../ops/ops';
import {ConfigDict, registerClass, Serializable, SerializableConstructor} from '../serialization';
import {Scalar, Variable} from '../tensor';
import {NamedVariableMap} from '../tensor_types';
import {Optimizer} from './optimizer';

export interface SGDOptimizerArgs{
  /**
   * TODO(cais): Add doc string.
   */
  learningRate?: number;

  /**
   * TODO(cais): Add doc string.
   */
  beta1?: number;

  /**
   * TODO(cais): Add doc string.
   */
  beta2?: number;

  /**
   * TODO(cais): Add doc string.
   */
  epsilon?: number;

  /**
   * TODO(cais): Add doc string.
   */
  decay?: number;

  /**
   * TODO(cais): Add doc string.
   */
  amsgrad?: boolean;
}

export class AdamOptimizer extends Optimizer {
  /** @nocollapse */
  static className = 'Adam';
  // This name must be compatible with keras and tf.keras in Python.

  private c: Scalar;
  private epsScalar: Scalar;
  private beta1Scalar: Scalar;
  private beta2Scalar: Scalar;
  private accBeta1: Variable;
  private accBeta2: Variable;
  private oneMinusBeta1: Scalar;
  private oneMinusBeta2: Scalar;
  private one: Scalar;

  // TODO(cais): Clean up.
  // private accumulatedFirstMoment: NamedVariableMap = {};
  // private accumulatedSecondMoment: NamedVariableMap = {};

  constructor(
      protected learningRate: number, protected beta1: number,
      protected beta2: number, protected epsilon: number = null) {
    super();
    this.c = keep(scalar(-learningRate));
    // b1, b2 keep initial value of beta* hyperparameters.
    this.beta1Scalar = keep(scalar(beta1));
    this.beta2Scalar = keep(scalar(beta2));
    tidy(() => {
      // accB* will be updated by batch.
      this.accBeta1 = scalar(beta1).variable();
      this.accBeta2 = scalar(beta2).variable();
    });
    this.oneMinusBeta1 = keep(scalar(1 - beta1));
    this.oneMinusBeta2 = keep(scalar(1 - beta2));
    this.one = keep(scalar(1));

    if (epsilon === null) {
      epsilon = ENV.get('EPSILON');
    }

    this.epsScalar = keep(scalar(epsilon));
  }

  private initializeWeights(variableGradients: NamedVariableMap) {
    console.log('In initializeWeights');  // DEBUG
    tidy(() => {
      const step = scalar(0).variable(false, null, 'int32');
      this.addWeight(step);

      const accumulatedFirstMoment: NamedVariableMap = {};
      const accumulatedSecondMoment: NamedVariableMap = {};
      for (const variableName in variableGradients) {
        const value = ENV.engine.registeredVariables[variableName];
        if (accumulatedFirstMoment[variableName] == null) {
          const trainable = false;
          accumulatedFirstMoment[variableName] =
              zerosLike(value).variable(trainable);
          accumulatedSecondMoment[variableName] =
              zerosLike(value).variable(trainable);
        }
      }
      for (const variableName in variableGradients) {
        this.addWeight(accumulatedFirstMoment[variableName]);
      }
      for (const variableName in variableGradients) {
        this.addWeight(accumulatedSecondMoment[variableName]);
      }
    });
  }

  // setWeights(weights: Tensor[]): void {
  //   super.setWeights(weights);
  //   dispose(this.accumulatedFirstMoment);
  //   dispose(this.accumulatedSecondMoment);
  // }

  applyGradients(variableGradients: NamedVariableMap) {
    if (this.weights == null) {
      this.initializeWeights(variableGradients);
    }

    tidy(() => {
      const oneMinusAccBeta1 = this.one.sub(this.accBeta1);
      const oneMinusAccBeta2 = this.one.sub(this.accBeta2);

      let i = 0;
      for (const variableName in variableGradients) {
        const value = ENV.engine.registeredVariables[variableName];        

        const gradient = variableGradients[variableName];
        // const firstMoment = this.accumulatedFirstMoment[variableName];
        // const secondMoment = this.accumulatedSecondMoment[variableName];
        const firstMoment = this.weights[i + 1];
        const secondMoment = this.weights[2 * i + 1];

        const newFirstMoment = this.beta1Scalar.mul(firstMoment)
                                   .add(this.oneMinusBeta1.mul(gradient));
        const newSecondMoment =
            this.beta2Scalar.mul(secondMoment)
                .add(this.oneMinusBeta2.mul(gradient.square()));

        const biasCorrectedFirstMoment = newFirstMoment.div(oneMinusAccBeta1);
        const biasCorrectedSecondMoment = newSecondMoment.div(oneMinusAccBeta2);

        // this.accumulatedFirstMoment[variableName].assign(newFirstMoment);
        // this.accumulatedSecondMoment[variableName].assign(newSecondMoment);
        firstMoment.assign(newFirstMoment);
        secondMoment.assign(newSecondMoment);

        const newValue =
            this.c
                .mul(biasCorrectedFirstMoment.div(
                    this.epsScalar.add(biasCorrectedSecondMoment.sqrt())))
                .add(value);
        value.assign(newValue);
      }

      this.accBeta1.assign(this.accBeta1.mul(this.beta1Scalar));
      this.accBeta2.assign(this.accBeta2.mul(this.beta2Scalar));

      const step = this.weights[0];
      step.assign(step.add(scalar(1, 'int32')));
      i++;
    });
  }

  dispose(): void {
    this.c.dispose();
    this.epsScalar.dispose();
    this.beta1Scalar.dispose();
    this.beta2Scalar.dispose();
    this.accBeta1.dispose();
    this.accBeta2.dispose();
    this.oneMinusBeta1.dispose();
    this.oneMinusBeta2.dispose();
    this.one.dispose();

    dispose(this.weights);
    // if (this.accumulatedFirstMoment != null) {
    //   Object.keys(this.accumulatedFirstMoment)
    //       .forEach(name => this.accumulatedFirstMoment[name].dispose());
    // }

    // if (this.accumulatedSecondMoment != null) {
    //   Object.keys(this.accumulatedSecondMoment)
    //       .forEach(name => this.accumulatedSecondMoment[name].dispose());
    // }
  }
  getConfig(): ConfigDict {
    return {
      learningRate: this.learningRate,
      beta1: this.beta1,
      beta2: this.beta2,
      epsilon: this.epsilon,
      decay: 0,
      amsrgad: false
    };
  }

  /** @nocollapse */
  static fromConfig<T extends Serializable>(
      cls: SerializableConstructor<T>, config: ConfigDict): T {
    if (config.decay != null && config.decay !== 0) {
      throw new Error('decay in AdamOptimizer is not implemented yet.');
    }
    if (config.amsgrad) {
      throw new Error('amsgrad in AdamOptimizer is not implemented yet.');
    }
    return new cls(
        config.learningRate, config.beta1, config.beta2, config.epsilon);
  }
}
registerClass(AdamOptimizer);
