import React from 'react';

const RunsList = (props) => {
  return (
    <div>
      <h2 className="title is-2">Submitted Runs:</h2>
      {
        props.runs.map((run) => {
          return (
            <h4
              key={run.id}
              className="box title is-4"
              style={
                {
                  backgroundColor: run.status === 0 ? 'lightblue' : (
                    run.status === 1 ? 'orange' : (
                      run.status === 2 ? 'lightgreen' : 'red'
                    )
                  )
                }
              }
            >{
              run.status === 0 ? "In Queue" : (
                run.status === 1 ? "Running..." : (
                  run.status === 2 ? "Completed" : "Failed"
                )
              )
            }
            </h4>
          )
        })
      }
    </div>
  )
};

export default RunsList;