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

import {IOHandler} from './io';

export type IORouter = (url: string) => IOHandler;

export class IORouterRegistry {
  // Singletone instance.
  private static instance: IORouterRegistry = null;

  private saveRouters: IORouter[];
  private loadRouters: IORouter[];

  private constructor() {
    this.saveRouters = [];
    this.loadRouters = [];
  }

  private static getInstance(): IORouterRegistry {
    if (IORouterRegistry.instance == null) {
      IORouterRegistry.instance = new IORouterRegistry();
    }
    return IORouterRegistry.instance;
  }

  /**
   * Register a save-handler router.
   *
   * @param saveRouter A function that maps a URL-like string onto an instance
   * of `IOHandler` with the `save` method defined or `null`.
   */
  static registerSaveRouter(saveRouter: IORouter) {
    IORouterRegistry.getInstance().saveRouters.push(saveRouter);
  }

  /**
   * Register a load-handler router.
   *
   * @param loadRouter A function that maps a URL-like string onto an instance
   * of `IOHandler` with the `load` method defined or `null`.
   */
  static registerLoadRouter(loadRouter: IORouter) {
    IORouterRegistry.getInstance().loadRouters.push(loadRouter);
  }

  /**
   * Look up IOHandler for saving, given a URL-like string.
   *
   * @param url
   * @returns If only one match is found, an instance of IOHandler with the
   * `save` method defined. If no match is found, `null`.
   * @throws Error, if more than one match is found.
   */
  static getSaveHandler(url: string): IOHandler {
    return IORouterRegistry.getHandler(url, 'save');
  }

  /**
   * Look up IOHandler for loading, given a URL-like string.
   *
   * @param url
   * @returns If only one match is found, an instance of IOHandler with the
   * `save` method defined. If no match is found, `null`.
   * @throws Error, if more than one match is found.
   */
  static getLoadHandler(url: string): IOHandler {
    return IORouterRegistry.getHandler(url, 'load');
  }

  private static getHandler(url: string, handlerType: 'save'|'load'):
      IOHandler {
    const validHandlers: IOHandler[] = [];
    for (const router of handlerType === 'load' ?
             this.getInstance().loadRouters :
             this.getInstance().saveRouters) {
      const handler = router(url);
      if (handler !== null) {
        validHandlers.push(handler);
      }
    }
    if (validHandlers.length === 0) {
      return null;
    } else if (validHandlers.length > 1) {
      throw new Error(
          `More than one (${validHandlers.length}) ${handlerType} handlers ` +
          ` are found for URL '${url}'.`);
    } else {
      return validHandlers[0];
    }
  }
}
