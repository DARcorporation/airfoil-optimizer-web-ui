import React from 'react';
import ReactDOM from 'react-dom';
import Collapsible from 'react-collapsible';

import AddRun from "./components/AddRun";
import RunsList from "./components/RunsList";

const App = (props) => {
  return (
    <section className="section">
      <div className="container">
        <div className="columns">
          <div className="column is-two-thirds">
            <Collapsible
              className="box"
              openedClassName="box"
              triggerClassName="title is-2"
              triggerOpenedClassName="title is-2"
              contentInnerClassName="box"
              trigger="Add New Run">
              <AddRun/>
            </Collapsible>
            <Collapsible
              className="box"
              openedClassName="box"
              triggerClassName="title is-2"
              triggerOpenedClassName="title is-2"
              contentInnerClassName="box"
              trigger="Run List">
              <RunsList/>
            </Collapsible>
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