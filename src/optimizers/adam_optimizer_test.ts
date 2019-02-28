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

import * as tf from '../index';
import {describeWithFlags} from '../jasmine_util';
import {scalar} from '../ops/ops';
import {ALL_ENVS, expectArraysClose} from '../test_util';

describeWithFlags('AdamOptimizer', ALL_ENVS, () => {
  it('basic', () => {
    const learningRate = .1;
    const beta1 = .8;
    const beta2 = .9;
    const optimizer = tf.train.adam(learningRate, beta1, beta2);

    const x = tf.tensor1d([2, 4]).variable();

    const f = () => x.square().sum() as tf.Scalar;

    let numTensors = tf.memory().numTensors;

    let cost = optimizer.minimize(f, /* returnCost */ true);

    // Cost & 2 accumulators should be the only additional arrays.
    expect(tf.memory().numTensors).toBe(numTensors + 3);
    // new_first_m = [
    //    beta1 * old_first_m_w1 + (1-beta1) * grad_w1,
    //    beta1 * old_first_m_w2 + (1-beta1) * grad_w2
    // ] = [.8, 1.6]
    // new_second_m = [
    //    beta2 * old_second_m_w1 + (1-beta2) * grad_w1**2,
    //    beta2 * old_second_m_w2 + (1-beta2) * grad_w2**2
    // ] = [1.6, 6.4]
    // m = [new_first_m/(1-acc_beta1)] = [4, 8]
    // v = [new_second_m/(1-acc_beta2)] = [16, 64]
    // x = [x - lr * m / sqrt(v)] = [1.9, 3.9]
    //
    expectArraysClose(x, [1.9, 3.9]);

    cost.dispose();
    numTensors = tf.memory().numTensors;

    cost = optimizer.minimize(f, /* returnCost */ false);

    // new_first_m = [
    //    beta1 * old_first_m_w1 + (1-beta1) * grad_w1,
    //    beta1 * old_first_m_w2 + (1-beta1) * grad_w2
    // ] = [1.4, 2.84]
    // new_second_m = [
    //    beta2 * old_second_m_w1 + (1-beta2) * grad_w1**2,
    //    beta2 * old_second_m_w2 + (1-beta2) * grad_w2**2
    // ] = [2.884, 11.884]
    // m = [new_first_m/(1-acc_beta1)] = [3.888888, 7.88889]
    // v = [new_second_m/(1-acc_beta2)] = [15.1789, 62.5473]
    // x = [x - lr * m / sqrt(v)] = [1.8000001, 3.8002]
    //
    expectArraysClose(x, [1.8000001, 3.8002]);
    // There should be no new additional Tensors.
    expect(tf.memory().numTensors).toBe(numTensors);

    expect(cost).toBe(null);

    x.dispose();
    optimizer.dispose();

    // The two tensors remaining should be the argument to variable()
    // and the step variable.
    expect(tf.memory().numTensors).toBe(2);
  });
  it('serialization round-trip', () => {
    const originalOpt = tf.train.adam(0.1, 0.2, 0.3, 2e-8);
    const reserialized =
        tf.AdamOptimizer.fromConfig(tf.AdamOptimizer, originalOpt.getConfig());
    expect(reserialized.getConfig()).toEqual(originalOpt.getConfig());
  });
  it('getWeights', () => {
    const learningRate = .1;
    const beta1 = .8;
    const beta2 = .9;
    const optimizer = tf.train.adam(learningRate, beta1, beta2);

    const x1 = tf.tensor1d([2, 4, 6]).variable();
    const x2 = tf.scalar(8).variable();
    const f = () => x1.add(x2).square().sum() as tf.Scalar;
    optimizer.minimize(f);
    optimizer.minimize(f);

    const weights = optimizer.getWeights();
    expect(weights.length).toEqual(5);  // TODO(cais): Is this right?
    // The number of times `minimize()` was called should be reflected
    // in the 1st (steps) variable.
    expectArraysClose(weights[0], scalar(2, 'int32'));
    // 1st momentum of variable x1.
    expect(weights[1].shape).toEqual([3]);
    // 1st momentum of variable x2.
    expect(weights[2].shape).toEqual([]);
    // 2nd momentum of variable x1.
    expect(weights[3].shape).toEqual([3]);
    // 2nd momentum of variable x2.
    expect(weights[4].shape).toEqual([]);
  });
  fit('setWeights', () => {
    console.log('==== BEGIN ====');  // DEBUG
    const learningRate = .1;
    const beta1 = .8;
    const beta2 = .9;
    const optimizer1 = tf.train.adam(learningRate, beta1, beta2);

    const x1 = tf.tensor1d([2, 4, 6]).variable();
    console.log('10 x1:');
    x1.print();

    const f1 = () => x1.square().sum() as tf.Scalar;

    // Call optimizer twice on f1.
    optimizer1.minimize(f1);
    optimizer1.minimize(f1);

    const x2 = x1.variable();
    const f2 = () => x2.square().sum() as tf.Scalar;

    // Set optimizer2's weights to the same as optimizer1.
    const weights1 = optimizer1.getWeights();
    expect(weights1.length).toEqual(3);  // Step and two moments.
    const optimizer2 = tf.train.adam(learningRate, beta1, beta2);
    // optimizer2.setWeights(weights1);

    const returnCost = true;
    const loss1 = optimizer1.minimize(f1, returnCost);
    loss1.print();  // DEBUG
  
    const loss2 = optimizer2.minimize(f2, returnCost);
    console.log('30 x1:');
    x1.print();
    console.log('30 x2:');
    x2.print();
    loss2.print();  // DEBUG

    expectArraysClose(loss1, loss2);
    // step count shold be the same.
    expectArraysClose(optimizer1.getWeights()[0], optimizer2.getWeights()[0]);
    // Weights should be the same.
    expectArraysClose(optimizer1.getWeights()[1], optimizer2.getWeights()[1]);
    expectArraysClose(optimizer1.getWeights()[2], optimizer2.getWeights()[2]);
    // optimizer1.getWeights()[1].print();
    // optimizer1.getWeights()[2].print();
    // optimizer2.getWeights()[1].print();
    // optimizer2.getWeights()[2].print();
  });
});
