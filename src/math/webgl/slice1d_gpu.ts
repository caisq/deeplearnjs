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

import {GPGPUContext} from './gpgpu_context';
import {GPGPUProgram} from './gpgpu_math';

export class Slice1DProgram implements GPGPUProgram {
  variableNames = ['source'];
  params: Array<{}>;
  outputShape: number[];
  userCode: string;

  // Caching uniform location for speed.
  startLoc: WebGLUniformLocation;

  constructor(destSize: number) {
    this.outputShape = [destSize];
    this.params = [];
    this.userCode = `
      uniform int start;

      void main() {
        int sourceIndex = start + getOutputCoords();
        setOutput(getSource(sourceIndex));
      }
    `;
  }

  getCustomSetupFunc(start: number) {
    return (gpgpu: GPGPUContext, webGLProgram: WebGLProgram) => {
      if (this.startLoc == null) {
        this.startLoc = gpgpu.getUniformLocation(webGLProgram, 'start');
      }
      gpgpu.gl.uniform1i(this.startLoc, start);
    };
  }
}