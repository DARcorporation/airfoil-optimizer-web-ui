import React from 'react';

const AddRun = (props) => {
  return (
    <form onSubmit={(event) => props.addRun(event)}>
      <h2 className="title is-2">Add a New Run</h2>
      <table>
        <col width="40%"/>
        <col width="20%"/>
        <col width="20%"/>
        <col width="20%"/>
        <tr>
          <td><h6 className="title is-6">Design Lift Coefficient</h6></td>
          <td>
            <div className="field">
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
          </td>
        </tr>
        <tr>
          <td><h6 className="title is-6">Number of CST coefficients</h6></td>
          <td>
            <div className="field">
              <input
                name="n_c"
                className="input"
                type="number"
                min="1"
                step="1"
                required
                value={props.n_c}
                onChange={props.handleChange}
              />
            </div>
          </td>
          <td>
            <div className="field">
              <input
                name="n_t"
                className="input"
                type="number"
                min="1"
                step="1"
                required
                value={props.n_t}
                onChange={props.handleChange}
              />
            </div>
          </td>
        </tr>
        <tr>
          <td><h6 className="title is-6">Fix the TE thickness?</h6></td>
          <td>
            <div className="field">
              <input
                name="fix_te"
                className=""
                type="checkbox"
                checked={props.fix_te}
                onChange={props.handleChange}
              />
            </div>
          </td>
        </tr>
        <tr>
          <td><h6 className="title is-6">Number of bits</h6></td>
          <td>
            <div className="field">
              <input
                name="b_c"
                className="input"
                type="number"
                min="1"
                step="1"
                required
                value={props.b_c}
                onChange={props.handleChange}
              />
            </div>
          </td>
          <td>
            <div className="field">
              <input
                name="b_t"
                className="input"
                type="number"
                min="1"
                step="1"
                required
                value={props.b_t}
                onChange={props.handleChange}
              />
            </div>
          </td>
          <td>
            <div className="field">
              <input
                name="b_te"
                className="input"
                type={props.fix_te ? "hidden" : "number"}
                min="1"
                step="1"
                required
                value={props.b_te}
                onChange={props.handleChange}
              />
            </div>
          </td>
        </tr>
        <tr>
          <td><h6 className="title is-6">Number of GA generations</h6></td>
          <td>
            <div className="field">
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
          </td>
        </tr>
        <tr>
          <td><h6 className="title is-6">Constraints</h6></td>
          <td>
            <div className="field">
              <label>
                <input
                  name="constrain_thickness"
                  className=""
                  type="checkbox"
                  checked={props.constrain_thickness}
                  onChange={props.handleChange}
                />
                {" minimal t/c"}
              </label>
            </div>
          </td>
          <td>
            <div className="field">
              <label>
                <input
                  name="constrain_area"
                  className=""
                  type="checkbox"
                  checked={props.constrain_area}
                  onChange={props.handleChange}
                />
                {" minimal area"}
              </label>
            </div>
          </td>
          <td>
            <div className="field">
              <label>
                <input
                  name="constrain_moment"
                  className=""
                  type="checkbox"
                  checked={props.constrain_moment}
                  onChange={props.handleChange}
                />
                {" limit moment"}
              </label>
            </div>
          </td>
        </tr>
        <tr>
          <td><h6 className="title is-6">Number of Processors</h6></td>
          <td>
            <div className="field">
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
          </td>
        </tr>
        <tr>
          <td><h6 className="title is-6">Report Results via Email?</h6></td>
          <td>
            <div className="field">
              <input
                name="report"
                className=""
                type="checkbox"
                checked={props.report}
                onChange={props.handleChange}
              />
            </div>
          </td>
        </tr>
      </table>
      <input
        type="submit"
        className="button is-primary is-large is-fullwidth"
        value="Submit"
      />
    </form>
  );
};

export default AddRun;