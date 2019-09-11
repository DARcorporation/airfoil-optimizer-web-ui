import axios from 'axios';
import React, {Component} from 'react';
import ReactDOM from 'react-dom';

import RunsList from "./components/RunsList";

class App extends Component {
  constructor(props) {
    super(props);
    this.state = {
      runs: [],
    }
  }

  componentDidMount() {
    this.getRuns();
  }

  getRuns() {
    axios.get(`${process.env.REACT_APP_RUNS_SERVICE_URL}/runs`)
      .then((res) => { this.setState({ runs: res.data.data.runs }); })
      .catch((err) => { console.log(err); })
  }

  render() {
    return (
      <section className="section">
        <div className="container">
          <div className="columns">
            <div className="column is-one-third">
              <br/>
              <h1 className="title is-1">All Runs</h1>
              <hr/><br/>
              <RunsList runs={this.state.runs}/>
            </div>
          </div>
        </div>
      </section>
    );
  }
}

ReactDOM.render(
  <App />,
  document.getElementById('root')
);