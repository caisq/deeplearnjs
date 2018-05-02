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

import {stringByteLength} from './io_utils';
import {IOHandler, ModelArtifacts, SaveResult} from './types';

const DATABASE_NAME = 'tensorflowjs';
const DATABASE_VERSION = 1;
const OBJECT_STORE_NAME = 'models_store';

export class BrowserIndexedDB implements IOHandler {
  protected readonly indexedDB: IDBFactory;
  protected readonly modelPath: string;

  constructor(modelPath: string) {
    if (!(window && window.indexedDB)) {
      // TODO(cais): Add more info about what IOHandler subtypes are available.
      //   Maybe point to a doc page on the web and/or automatically determine
      //   the available IOHandlers and print them in the error message.
      throw new Error('The current environment does not support IndexedDB.');
    }
    this.indexedDB = window.indexedDB;

    if (modelPath == null || !modelPath) {
      throw new Error(
          'For IndexedDB, modelPath must not be null, undefined or empty.');
    }
    this.modelPath = modelPath;
  }

  async save(modelArtifacts: ModelArtifacts): Promise<SaveResult> {
    // TODO(cais): Support saving GraphDef models.
    if (modelArtifacts.modelTopology instanceof ArrayBuffer) {
      throw new Error(
          'BrowserLocalStorage.save() does not support saving model topology ' +
          'in binary formats yet.');
    }

    return new Promise<SaveResult>((resolve, reject) => {
      const openRequest = this.indexedDB.open(DATABASE_NAME, DATABASE_VERSION);

      // Create the schema.
      openRequest.onupgradeneeded = () => {
        const db = openRequest.result as IDBDatabase;
        db.createObjectStore(OBJECT_STORE_NAME, {keyPath: 'modelPath'});
      };

      openRequest.onsuccess = () => {
        console.log('openRequest.onsuccess');  // DEBUG
        const db = openRequest.result as IDBDatabase;
        const tx = db.transaction(OBJECT_STORE_NAME, 'readwrite');
        const store = tx.objectStore(OBJECT_STORE_NAME);

        const putRequest =
            store.put({modelPath: this.modelPath, modelArtifacts});
        putRequest.onsuccess = () => {
          console.log('putRequest.onsuccess');  // DEBUG
          resolve({
            modelArtifactsInfo: {
              dateSaved: new Date(),
              modelTopologyType: 'KerasJSON',
              modelTopologyBytes: stringByteLength(
                  JSON.stringify(modelArtifacts.modelTopology)),
              weightSpecsBytes:
                  stringByteLength(JSON.stringify(modelArtifacts.weightSpecs)),
              weightDataBytes: modelArtifacts.weightData.byteLength,
            },
          });
        };
        putRequest.onerror = (error) => {
          reject(error);
        };
        tx.oncomplete = () => {
          db.close();
        };
      };
      openRequest.onerror = (error) => {
        reject(error);
      };
    });
  }

  async load(): Promise<ModelArtifacts> {
    return new Promise<ModelArtifacts>((resolve, reject) => {
      const openRequest = this.indexedDB.open(DATABASE_NAME, DATABASE_VERSION);

      openRequest.onupgradeneeded = () => {
        const db = openRequest.result as IDBDatabase;
        db.createObjectStore(OBJECT_STORE_NAME, {keyPath: 'modelPath'});
      };

      openRequest.onsuccess = () => {
        const db = openRequest.result as IDBDatabase;
        const tx = db.transaction(OBJECT_STORE_NAME, 'readwrite');
        const store = tx.objectStore(OBJECT_STORE_NAME);

        const getRequest = store.get(this.modelPath);
        getRequest.onsuccess = () => {
          resolve(getRequest.result.modelArtifacts);
        };
        getRequest.onerror = (error) => {
          reject(error);
        };
        tx.oncomplete = () => {
          db.close();
        };
      };
      openRequest.onerror = (error) => {
        reject(error);
      };
    });
  }
}

// TODO(cais): Doc string and code snippet.
export function browserIndexedDB(modelPath: string): BrowserIndexedDB {
  return new BrowserIndexedDB(modelPath);
}
