/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React from "react"
import ReactDOM from "react-dom"

import {
  ComponentProps,
  withStreamlitConnection,
} from "streamlit-component-lib"


import ContributionGraph from "./ContributionGraph"
import Selector from "./Selector"

const LlmViewerComponent = (props: ComponentProps) => {
  switch (props.args['component']) {
    case 'graph':
      return <ContributionGraph />
    case 'selector':
      return <Selector />
    default:
      return <></>
  }
};

const StreamlitLlmViewerComponent = withStreamlitConnection(LlmViewerComponent)

ReactDOM.render(
  <React.StrictMode>
    <StreamlitLlmViewerComponent />
  </React.StrictMode>,
  document.getElementById("root")
)
