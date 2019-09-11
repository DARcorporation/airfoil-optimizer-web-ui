import React from 'react';

const RunsList = (props) => {
  return (
    <div>
      {
        props.runs.map((run) => {
          return (
            <h4
              key={run.id}
              className="box title is-4"
            >{run.cl}, {run.n_c}, {run.n_t}, {run.b_c}, {run.b_t}, {run.b_te}, {run.gen}
            </h4>
          )
        })
      }
    </div>
  )
};

export default RunsList;