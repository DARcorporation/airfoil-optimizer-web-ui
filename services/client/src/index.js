import axios from 'axios';
import React, {Component} from 'react';
import ReactDOM from 'react-dom';

import AddRun from "./components/AddRun";
import RunsList from "./components/RunsList";

class App extends Component {
  constructor(props) {
    super(props);
    this.default_state = {
      runs: [],
      cl: 0.0,
      n_c: 3,
      n_t: 3,
      fix_te: true,
      gen: 0,
      tolx: 1e-8,
      tolf: 1e-8,
      constrain_thickness: true,
      constrain_area: true,
      constrain_moment: false,
      n_proc: 1,
      report: false,
    };
    this.state = this.default_state;
    this.addRun = this.addRun.bind(this);
    this.handleChange = this.handleChange.bind(this);
  }

  componentDidMount() {
    this.interval = setInterval(() => this.getRuns(), 1000);
  }

  componentWillUnmount() {
    clearInterval(this.interval);
  }

  addRun(event) {
    event.preventDefault();
    const data = {
      cl: this.state.cl,
      n_c: this.state.n_c,
      n_t: this.state.n_t,
      fix_te: this.fix_te,
      gen: this.state.gen,
      tolx: this.state.tolx,
      tolf: this.state.tolf,
      constrain_thickness: this.state.constrain_thickness,
      constrain_area: this.state.constrain_area,
      constrain_moment: this.state.constrain_moment,
      n_proc: this.state.n_proc,
      report: this.state.report,
    };
    axios.post(`${process.env.REACT_APP_RUNS_SERVICE_URL}/runs`, data)
      .then((res) => {
        this.getRuns();
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
    if (['fix_te', 'constrain_thickness', 'constrain_area', 'constrain_moment', 'report'].includes(event.target.name)) {
      obj[event.target.name] = event.target.checked;
    } else {
      obj[event.target.name] = event.target.value;
    }
    this.setState(obj);
  };

  render() {
    return (
      <section className="section">
        <div className="container">
          <div className="columns">
            <div className="column is-two-thirds">
              <AddRun
                cl={this.state.cl}
                n_c={this.state.n_c}
                n_t={this.state.n_t}
                fix_te={this.state.fix_te}
                gen={this.state.gen}
                tolx={this.state.tolx}
                tolf={this.state.tolf}
                constrain_thickness={this.state.constrain_thickness}
                constrain_area={this.state.constrain_area}
                constrain_moment={this.state.constrain_moment}
                n_proc={this.state.n_proc}
                report={this.state.report}
                addRun={this.addRun}
                handleChange={this.handleChange}
              />
              <hr/><br/><br/>
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