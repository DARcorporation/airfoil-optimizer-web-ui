import React from 'react';
import ReactDOM from 'react-dom';
import Collapsible from 'react-collapsible';

import AddRun from "./components/AddRun";
import RunsList from "./components/RunsList";

const MyCollapsible = (props) => {
  return (
    <Collapsible
      className="box"
      openedClassName="box"
      triggerClassName="title is-2"
      triggerOpenedClassName="title is-2"
      contentInnerClassName="box"
      trigger={props.title}>
      {props.children}
    </Collapsible>
  )
};

const App = (props) => {
  return (
    <section className="section">
      <div className="container">
        <div className="columns">
          <div className="column is-two-thirds">
            <MyCollapsible title="Add New Run">
              <AddRun/>
            </MyCollapsible>
            <MyCollapsible title="Run List">
              <RunsList/>
            </MyCollapsible>
          </div>
        </div>
      </div>
    </section>
  );
};

ReactDOM.render(
  <App />,
  document.getElementById('root')
);