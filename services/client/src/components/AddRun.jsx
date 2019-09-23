import React, { Component } from 'react';
import axios from "axios";

const input_style = {
  invalid: {
    border: '1px solid red'
  }
};

class AddRun extends Component {
  constructor(props) {
    super(props);
    this.default_state = {
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
    this.state = {
      ...this.default_state,
      collapsed: true,
    };
    this.collapseHandle = this.collapseHandle.bind(this);
    this.addRun = this.addRun.bind(this);
    this.handleChange = this.handleChange.bind(this);
  }

  collapseHandle(event) {
    this.setState({collapsed: !this.state.collapsed});
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

      })
      .catch((err) => { console.log(err); });
  };

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
      <form onSubmit={(event) => this.addRun(event)}>
        <div className="box">
          <h6 className="title is-6">Basic Problem Setup</h6>
          <div className="columns">
            <div className="column">
              Design Lift Coefficient:<br/>
              <input
                name="cl"
                className="input"
                type="number"
                step="0.01"
                required
                value={this.state.cl}
                onChange={this.handleChange}
              />
            </div>
            <div className="column">
              Fix the TE Thickness?<br/>
              <input
                name="fix_te"
                className=""
                type="checkbox"
                checked={this.state.fix_te}
                onChange={this.handleChange}
              />
            </div>
          </div>
          <div className="columns">
            <div className="column">
              Constrain the t/c?<br/>
              <input
                name="constrain_thickness"
                type="checkbox"
                checked={this.state.constrain_thickness}
                onChange={this.state.handleChange}
              />
            </div>
            <div className="column">
              Constrain the area<br/>
              <input
                name="constrain_area"
                type="checkbox"
                checked={this.state.constrain_area}
                onChange={this.handleChange}
              />
            </div>
            <div className="column">
              Constrain the moment?<br/>
              <input
                name="constrain_moment"
                type="checkbox"
                checked={this.state.constrain_moment}
                onChange={this.handleChange}
              />
            </div>
          </div>
        </div>
        <div className="box">
          <h6 className="title is-6">Numbers of Design Variables</h6>
          <div className="columns">
            <div className="column">
              Chord Line
              <input
                name="n_c"
                className="input"
                type="number"
                min="0"
                step="1"
                required
                value={this.state.n_c}
                onChange={this.handleChange}
              />
            </div>
            <div className="column">
              Thickness Distribution
              <input
                name="n_t"
                className="input"
                type="number"
                min="0"
                step="1"
                required
                value={this.state.n_t}
                onChange={this.handleChange}
              />
            </div>
          </div>
          <i className="italic">It is recommended to use between 3 and 6 coefficients for each variable.</i>
        </div>
        <div className="box">
          <h6 className="title is-6">Termination Settings</h6>
          <div className="columns">
            <div className="column">
              Number of Generations
              <input
                name="gen"
                className="input"
                type="number"
                min="0"
                step="1"
                required
                value={this.state.gen}
                onChange={this.handleChange}
              />
            </div>
            <div className="column">
              Design Vector Tolerance
              <input
                name="tolx"
                className="input"
                type="text"
                pattern="[+-]?([1-9]\d*|0)?(\.\d*)?([Ee][+-]?\d+)?"
                required
                style={input_style}
                value={this.state.tolx}
                onChange={this.handleChange}
              />
            </div>
            <div className="column">
              Objective Function Tolerance
              <input
                name="tolf"
                className="input"
                type="text"
                pattern="[+-]?([1-9]\d*|0)?(\.\d*)?([Ee][+-]?\d+)?"
                required
                value={this.state.tolf}
                onChange={this.handleChange}
              />
            </div>
          </div>
        </div>
        <div className="box">
          <h6 className="title is-6">Miscellaneous Settings</h6>
          <div className="columns">
            <div className="column">
              Number of Processors:<br/>
              <input
                name="n_proc"
                className="input"
                type="number"
                step="1"
                min="1"
                required
                value={this.state.n_proc}
                onChange={this.handleChange}
              />
            </div>

            <div className="column">
              Report Result via Email?<br/>
              <input
                name="report"
                className=""
                type="checkbox"
                checked={this.state.report}
                onChange={this.handleChange}
              />
            </div>
          </div>
        </div>
        <input
          type="submit"
          className="button is-primary is-large is-fullwidth"
          value="Submit"
        />
      </form>
    );
  }
}

export default AddRun;