import React, { Component } from 'react';
import axios from "axios";

class RunsList extends Component {
  constructor(props) {
    super(props);
    this.state = {
      runs: [],
    }
  }

  componentDidMount() {
    this.interval = setInterval(() => this.getRuns(), 1000);
  }

  componentWillUnmount() {
    clearInterval(this.interval);
  }

  getRuns() {
    axios.get(`${process.env.REACT_APP_RUNS_SERVICE_URL}/runs`)
      .then((res) => { this.setState({ runs: res.data.data.runs }); })
      .catch((err) => { console.log(err); })
  }

  render() {
    return (
      <div>
        {
          this.state.runs.map((run) => {
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
    );
  }
}

export default RunsList;