import React, { Component } from 'react';
import axios from "axios";
import MyCollapsible from "./MyCollapsible";

const input_style = {
  invalid: {
    border: '1px solid red'
  }
};

class AddRun extends Component {
  constructor(props) {
    super(props);
    this.default_state = {
      cl: 1.0,
      n_c: 3,
      n_t: 3,
      fix_te: true,
      gen: 1000,
      tolx: 1e-8,
      tolf: 1e-8,
      constrain_thickness: true,
      constrain_area: true,
      constrain_moment: false,
      n_proc: 28,
      report: false,
    };
    this.state = {
      ...this.default_state,
      collapsed: true,
    };
    this.collapseHandle = this.collapseHandle.bind(this);
    this.addRun = this.addRun.bind(this);
    this.handleChange = this.handleChange.bind(this);
    this.radioHelper = this.radioHelper.bind(this);
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
      fix_te: this.state.fix_te,
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
      obj[event.target.name] = (event.target.value === 'true')
    } else {
      obj[event.target.name] = event.target.value;
    }
    this.setState(obj);
  };

  radioHelper(event) {
    this.handleChange(event);
  }


  render() {
    return (
      <form onSubmit={(event) => this.addRun(event)}>

        <MyCollapsible titleClassName="title is-4" title="Basic Problem Setup">
          <div className="columns">
            <div className="field column">
              <label className="label">Design Lift Coefficient</label>
              <div className="control">
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
            </div>
            <div className="column"/>
          </div>

          <div className="columns">
            <div className="field column">
              <label className="label">Fix TE Thickness?</label>
              <div className="field">
                <input className="is-checkradio"
                       type="radio"
                       name="fix_te"
                       value={true}
                       checked={this.state.fix_te}
                       onChange={this.handleChange}
                />
                <label
                  className="radio"
                  onClick={(event) => this.radioHelper(
                    {
                      target: {
                        name: "fix_te",
                        value: "true",
                      }
                    })}
                >Yes</label>
                <input className="is-checkradio"
                       type="radio"
                       name="fix_te"
                       value={false}
                       checked={!this.state.fix_te}
                       onChange={this.handleChange}
                />
                <label
                  className="radio"
                  onClick={(event) => this.radioHelper(
                    {
                      target: {
                        name: "fix_te",
                        value: "false",
                      }
                    })}
                >No</label>
              </div>
            </div>

            <div className="field column">
              <label className="label">Constrain Thickness?</label>
              <div className="control">
                <input className="is-checkradio"
                       type="radio"
                       name="constrain_thickness"
                       value={true}
                       checked={this.state.constrain_thickness}
                       onChange={this.handleChange}
                />
                <label
                  className="radio"
                  onClick={(event) => this.radioHelper(
                    {
                      target: {
                        name: "constrain_thickness",
                        value: "true",
                      }
                    })}
                >Yes</label>
                <input className="is-checkradio"
                       type="radio"
                       name="constrain_thickness"
                       value={false}
                       checked={!this.state.constrain_thickness}
                       onChange={this.handleChange}
                />
                <label
                  className="radio"
                  onClick={(event) => this.radioHelper(
                    {
                      target: {
                        name: "constrain_thickness",
                        value: "false",
                      }
                    })}
                >No</label>
              </div>
            </div>
          </div>

          <div className="columns">
            <div className="field column">
              <label className="label">Constrain Area?</label>
              <div className="control">
                <input className="is-checkradio"
                       type="radio"
                       name="constrain_area"
                       value={true}
                       checked={this.state.constrain_area}
                       onChange={this.handleChange}
                />
                <label
                  className="radio"
                  onClick={(event) => this.radioHelper(
                    {
                      target: {
                        name: "constrain_area",
                        value: "true",
                      }
                    })}
                >No</label>
                <input className="is-checkradio"
                       type="radio"
                       name="constrain_area"
                       value={false}
                       checked={!this.state.constrain_area}
                       onChange={this.handleChange}
                />
                <label
                  className="radio"
                  onClick={(event) => this.radioHelper(
                    {
                      target: {
                        name: "constrain_area",
                        value: "false",
                      }
                    })}
                >No</label>
              </div>
            </div>

            <div className="field column">
              <label className="label">Constrain Moment?</label>
              <div className="control">
                <input className="is-checkradio"
                       type="radio"
                       name="constrain_moment"
                       value={true}
                       checked={this.state.constrain_moment}
                       onChange={this.handleChange}
                />
                <label
                  className="radio"
                  onClick={(event) => this.radioHelper(
                    {
                      target: {
                        name: "constrain_moment",
                        value: "true",
                      }
                    })}
                >Yes</label>
                <input className="is-checkradio"
                       type="radio"
                       name="constrain_moment"
                       value={false}
                       checked={!this.state.constrain_moment}
                       onChange={this.handleChange}
                />
                <label
                  className="radio"
                  onClick={(event) => this.radioHelper(
                    {
                      target: {
                        name: "constrain_moment",
                        value: "false",
                      }
                    })}
                >No</label>
              </div>
            </div>
          </div>
        </MyCollapsible>

        <MyCollapsible titleClassName="title is-4" title="Number of Design Variables">
          <div className="columns">
            <div className="field column">
              <label className="label">Chord Line</label>
              <div className="control">
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
            </div>

            <div className="field column">
              <label className="label">Thickness Distribution</label>
              <div className="control">
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
          </div>
          <i className="italic">It is recommended to use between 3 and 6 for each.</i>
        </MyCollapsible>

        <MyCollapsible titleClassName="title is-4" title="Termination Settings">
          <div className="columns">
            <div className="field column">
              <label className="label">Number of Generations</label>
              <div className="control">
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
            </div>
            <div className="column"/>
          </div>
          <div className="columns">
            <div className="field column">
              <label className="label">Design Vector Tolerance</label>
              <div className="control">
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
            </div>
            <div className="field column">
              <label className="label">Objective Function Tolerance</label>
              <div className="control">
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
        </MyCollapsible>

        <MyCollapsible titleClassName="title is-4" title="Miscellaneous Settings">
          <div className="columns">
            <div className="field column">
              <label className="label">Number of Processors</label>
              <div className="control">
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
            </div>

            <div className="field column">
              <label className="label">Report Result via Email?</label>
              <div className="control">
                <input className="is-checkradio"
                       type="radio"
                       name="report"
                       value={true}
                       checked={this.state.report}
                       onChange={this.handleChange}
                />
                <label
                  className="radio"
                  onClick={(event) => this.radioHelper(
                    {
                      target: {
                        name: "report",
                        value: "true",
                      }
                    })}
                >Yes</label>
                <input className="is-checkradio"
                       type="radio"
                       name="report"
                       value={false}
                       checked={!this.state.report}
                       onChange={this.handleChange}
                />
                <label
                  className="radio"
                  onClick={(event) => this.radioHelper(
                    {
                      target: {
                        name: "report",
                        value: "false",
                      }
                    })}
                >No</label>
              </div>
            </div>
          </div>
        </MyCollapsible>
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