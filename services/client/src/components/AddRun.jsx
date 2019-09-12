import React from 'react';

const AddRun = (props) => {
  return (
    <form onSubmit={(event) => props.addRun(event)}>
      <div className="field">
        <input
          name="cl"
          className="input is-large"
          type="number"
          placeholder="Enter the design lift coefficient"
          required
          value={props.cl}
          onChange={props.handleChange}
        />
      </div>
      <div className="field">
        <input
          name="n_c"
          className="input is-large"
          type="number"
          placeholder="Enter the number of CST coefficients for the chord line"
          required
          value={props.n_c}
          onChange={props.handleChange}
        />
      </div>
      <div className="field">
        <input
          name="n_t"
          className="input is-large"
          type="number"
          placeholder="Enter the number of CST coefficients for the thickness distribution"
          required
          value={props.n_t}
          onChange={props.handleChange}
        />
      </div>
      <div className="field">
        <input
          name="b_c"
          className="input is-large"
          type="number"
          placeholder="Enter the number of bits per chord line CST coefficient"
          required
          value={props.b_c}
          onChange={props.handleChange}
        />
      </div>
      <div className="field">
        <input
          name="b_t"
          className="input is-large"
          type="number"
          placeholder="Enter the number of bits per thickness CST coefficient"
          required
          value={props.b_t}
          onChange={props.handleChange}
        />
      </div>
      <div className="field">
        <input
          name="b_te"
          className="input is-large"
          type="number"
          placeholder="Enter the number of bits for the TE thickness"
          required
          value={props.b_te}
          onChange={props.handleChange}
        />
      </div>
      <div className="field">
        <input
          name="gen"
          className="input is-large"
          type="number"
          placeholder="Enter the number of generations for the genetic algorithm"
          required
          value={props.gen}
          onChange={props.handleChange}
        />
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