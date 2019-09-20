import React from 'react';

const AddRun = (props) => {
  return (
    <form onSubmit={(event) => props.addRun(event)} className="box">
      <h2 className="title is-2">Add a New Run</h2>
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
              value={props.cl}
              onChange={props.handleChange}
            />
          </div>
          <div className="column">
            GA Generations:
            <input
              name="gen"
              className="input"
              type="number"
              min="0"
              step="1"
              required
              value={props.gen}
              onChange={props.handleChange}
            />
          </div>
          <div className="column">
            Fix the TE Thickness?<br/>
            <input
              name="fix_te"
              className=""
              type="checkbox"
              checked={props.fix_te}
              onChange={props.handleChange}
            />
          </div>
        </div>
        <div className="columns">
          <div className="column">
            Constrain the t/c?<br/>
            <input
              name="constrain_thickness"
              type="checkbox"
              checked={props.constrain_thickness}
              onChange={props.handleChange}
            />
          </div>
          <div className="column">
            Constrain the area<br/>
            <input
              name="constrain_area"
              type="checkbox"
              checked={props.constrain_area}
              onChange={props.handleChange}
            />
          </div>
          <div className="column">
            Constrain the moment?<br/>
            <input
              name="constrain_area"
              type="checkbox"
              checked={props.constrain_moment}
              onChange={props.handleChange}
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
              step="0.01"
              required
              value={props.n_c}
              onChange={props.handleChange}
            />
          </div>
          <div className="column">
            Thickness Distribution
            <input
              name="n_t"
              className="input"
              type="number"
              step="0.01"
              required
              value={props.n_t}
              onChange={props.handleChange}
            />
          </div>
        </div>
        <i className="italic">It is recommended to use between 3 and 6 coefficients for each variable.</i>
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
              value={props.n_proc}
              onChange={props.handleChange}
            />
          </div>

          <div className="column">
            Report Result via Email?<br/>
            <input
              name="report"
              className=""
              type="checkbox"
              checked={props.report}
              onChange={props.handleChange}
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
};

export default AddRun;