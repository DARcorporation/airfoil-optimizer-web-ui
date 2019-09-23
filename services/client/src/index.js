import React from 'react';
import ReactDOM from 'react-dom';
import "./index.scss"

import AddRun from "./components/AddRun";
import RunsList from "./components/RunsList";
import MyCollapsible from "./components/MyCollapsible";


const App = (props) => {
  return (
    <section className="section">
      <div className="container">
        <div className="columns">
          <div className="column is-two-thirds">
            <MyCollapsible titleClassName="title is-2" title="Add New Run">
              <AddRun/>
            </MyCollapsible>
            <MyCollapsible titleClassName="title is-2" title="Run List">
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