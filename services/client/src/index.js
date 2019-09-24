import React from 'react';
import ReactDOM from 'react-dom';
import "./index.scss"

import AddRun from "./components/AddRun";
import RunsList from "./components/RunsList";
import MyCollapsible from "./components/MyCollapsible";
import {Button} from "@material-ui/core";

const App = (props) => {
  const [addRunOpen, setAddRunOpen] = React.useState(false);

  const handleClickOpen = () => {
    setAddRunOpen(true);
  };

  const handleClose = () => {
    setAddRunOpen(false);
  };

  return (
    <section className="section">
      <div className="container">
        <div className="columns">
          <div className="column is-two-thirds">
            <Button
              variant="contained"
              color="primary"
              onClick={handleClickOpen}
            >Submit New Run</Button>
            <AddRun open={addRunOpen} onClose={handleClose}/>
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