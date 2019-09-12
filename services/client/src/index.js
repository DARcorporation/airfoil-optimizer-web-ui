import axios from 'axios';
import React, {Component} from 'react';
import ReactDOM from 'react-dom';

import AddRun from "./components/AddRun";
import RunsList from "./components/RunsList";

class App extends Component {
  constructor(props) {
    super(props);
    this.state = {
      runs: [],
      cl: 0,
      n_c: 0,
      n_t: 0,
      b_c: 0,
      b_t: 0,
      b_te: 0,
      gen: 0,
    };
    this.addRun = this.addRun.bind(this);
    this.handleChange = this.handleChange.bind(this);
  }

  componentDidMount() {
    this.getRuns();
  }

  addRun(event) {
    event.preventDefault();
    const data = {
      cl: this.state.cl,
      n_c: this.state.n_c,
      n_t: this.state.n_t,
      b_c: this.state.b_c,
      b_t: this.state.b_t,
      b_te: this.state.b_te,
      gen: this.state.gen,
    };
    axios.post(`${process.env.REACT_APP_RUNS_SERVICE_URL}/runs`, data)
      .then((res) => {
        this.getRuns();
        this.setState({
          cl: 0,
          n_c: 0,
          n_t: 0,
          b_c: 0,
          b_t: 0,
          b_te: 0,
          gen: 0,
        })
      })
      .catch((err) => { console.log(err); });
  };

  getRuns() {
    axios.get(`${process.env.REACT_APP_RUNS_SERVICE_URL}/runs`)
      .then((res) => { this.setState({ runs: res.data.data.runs }); })
      .catch((err) => { console.log(err); })
  }

  handleChange(event) {
    const obj = {};
    obj[event.target.name] = event.target.value;
    this.setState(obj);
  };

  render() {
    return (
      <section className="section">
        <div className="container">
          <div className="columns">
            <div className="column is-half">
              <br/>
              <h1 className="title is-1">All Runs</h1>
              <hr/><br/>
              <AddRun
                cl={this.state.cl}
                n_c={this.state.n_c}
                n_t={this.state.n_t}
                b_c={this.state.b_c}
                b_t={this.state.b_t}
                b_te={this.state.b_te}
                gen={this.state.gen}
                addRun={this.addRun}
                handleChange={this.handleChange}
              />
              <br/><br/>
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