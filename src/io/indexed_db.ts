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

// tslint:disable:max-line-length
import {stringByteLength} from './io_utils';
import {IOHandler, ModelArtifacts, ModelArtifactsInfo, SaveResult} from './types';
// tslint:enable:max-line-length

const DATABASE_NAME = 'tensorflowjs';
const DATABASE_VERSION = 1;
const OBJECT_STORE_NAME = 'models_store';

/**
 * Delete the entire database for tensorfowjs, including the models store.
 */
export async function deleteDatabase(): Promise<void> {
  const idbFactory = getIndexedDBFactory();

  return new Promise<void>((resolve, reject) => {
    const deleteRequest = idbFactory.deleteDatabase(DATABASE_NAME);
    deleteRequest.onsuccess = () => {
      resolve();
    };
    deleteRequest.onerror = (error) => {
      reject();
    };
  });
}

function getIndexedDBFactory(): IDBFactory {
  if (window == null) {
    // TODO(cais): Add more info about what IOHandler subtypes are available.
    //   Maybe point to a doc page on the web and/or automatically determine
    //   the available IOHandlers and print them in the error message.
    throw new Error(
        'Failed to obtain IndexedDB factory because the current environment' +
        'is not a web browser.');
  }
  // tslint:disable-next-line:no-any
  const theWindow: any = window;
  return theWindow.indexedDB || theWindow.mozIndexedDB ||
      theWindow.webkitIndexedDB || theWindow.msIndexedDB ||
      theWindow.shimIndexedDB;
}

export class BrowserIndexedDB implements IOHandler {
  protected readonly indexedDB: IDBFactory;
  protected readonly modelPath: string;

  constructor(modelPath: string) {
    this.indexedDB = getIndexedDBFactory();

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
      const modelArtifactsInfo: ModelArtifactsInfo = {
        dateSaved: new Date(),
        modelTopologyType: 'KerasJSON',
        modelTopologyBytes:
            stringByteLength(JSON.stringify(modelArtifacts.modelTopology)),
        weightSpecsBytes:
            stringByteLength(JSON.stringify(modelArtifacts.weightSpecs)),
        weightDataBytes: modelArtifacts.weightData.byteLength,
      };

      const openRequest = this.indexedDB.open(DATABASE_NAME, DATABASE_VERSION);

      // Create the schema.
      openRequest.onupgradeneeded = () => {
        this.setUpDatabase(openRequest);
      };

      openRequest.onsuccess = () => {
        const db = openRequest.result as IDBDatabase;
        const tx = db.transaction(OBJECT_STORE_NAME, 'readwrite');
        const store = tx.objectStore(OBJECT_STORE_NAME);

        const putRequest = store.put(
            {modelPath: this.modelPath, modelArtifacts, modelArtifactsInfo});
        putRequest.onsuccess = () => {
          resolve({modelArtifactsInfo});
        };
        putRequest.onerror = (error) => {
          reject(putRequest.error);
        };
        tx.oncomplete = () => {
          db.close();
        };
      };
      openRequest.onerror = (error) => {
        reject(openRequest.error);
      };
    });
  }

  async load(): Promise<ModelArtifacts> {
    return new Promise<ModelArtifacts>((resolve, reject) => {
      const openRequest = this.indexedDB.open(DATABASE_NAME, DATABASE_VERSION);

      openRequest.onupgradeneeded = () => {
        this.setUpDatabase(openRequest);
      };

      openRequest.onsuccess = () => {
        const db = openRequest.result as IDBDatabase;
        const tx = db.transaction(OBJECT_STORE_NAME, 'readwrite');
        const store = tx.objectStore(OBJECT_STORE_NAME);

        const getRequest = store.get(this.modelPath);
        getRequest.onsuccess = () => {
          if (getRequest.result === undefined) {
            reject(new Error(
                `Cannot find model with path '${this.modelPath}' ` +
                `in IndexedDB.`));
          } else {
            resolve(getRequest.result.modelArtifacts);
          }
        };
        getRequest.onerror = (error) => {
          reject(getRequest.error);
        };
        tx.oncomplete = () => {
          db.close();
        };
      };
      openRequest.onerror = (error) => {
        reject(openRequest.error);
      };
    });
  }

  private setUpDatabase(openRequest: IDBRequest) {
    const db = openRequest.result as IDBDatabase;
    db.createObjectStore(OBJECT_STORE_NAME, {keyPath: 'modelPath'});
  }
}

/**
 * Factory function for browser IndexedDB IOHandler.
 *
 * ```js
 * const model = tf.sequential();
 * model.add(
 *     tf.layers.dense({units: 1, inputShape: [100], activation: 'sigmoid'}));
 *
 * const saveResult = await model.save(tf.io.browserIndexedDB('MyModel'));
 * console.log(saveResult);
 * ```
 *
 * @param modelPath A unique identifier for the model to be saved. Must be a
 *   non-empty string.
 * @returns An instance of `BrowserIndexedDB` (sublcass of `IOHandler`),
 *   which can be used with, e.g., `tf.Model.save`.
 */
export function browserIndexedDB(modelPath: string): BrowserIndexedDB {
  return new BrowserIndexedDB(modelPath);
}
