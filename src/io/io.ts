/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import {files, triggerDownloads} from './files';
import {browserIndexedDB} from './indexed_db';
import {decodeWeights, encodeWeights} from './io_utils';
import {browserLocalStorage} from './local_storage';
// tslint:disable-next-line:max-line-length
import {IOHandler, LoadHandler, ModelArtifacts, SaveConfig, SaveHandler, SaveResult, WeightsManifestConfig, WeightsManifestEntry} from './types';
import {loadWeights} from './weights_loader';

export {
  browserIndexedDB,
  browserLocalStorage,
  decodeWeights,
  encodeWeights,
  files,
  IOHandler,
  LoadHandler,
  loadWeights,
  ModelArtifacts,
  SaveConfig,
  SaveHandler,
  SaveResult,
  triggerDownloads,
  WeightsManifestConfig,
  WeightsManifestEntry
};
