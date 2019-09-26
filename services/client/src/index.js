import React, {useEffect} from "react";
import ReactDOM from "react-dom";
import {
  AppBar,
  createMuiTheme,
  Fab,
  makeStyles,
  Toolbar,
  Typography,
  Grid,
  useTheme,
} from "@material-ui/core";
import { ThemeProvider } from "@material-ui/styles";
import AddIcon from "@material-ui/icons/Add";
import axios from "axios";

import AddRun from "./components/AddRun";
import Run from "./components/Run";
import "./index.scss"

const useStyles = makeStyles(theme => ({
  root: {
    margin: 0,
    padding: theme.spacing(1),
  },
  fab: {
    position: "fixed",
    bottom: theme.spacing(2),
    right: theme.spacing(2),
  },
  appBar: {
    top: 0,
    bottom: "auto",
  },
}));

const App = (props) => {
  const theme = useTheme();
  const classes = useStyles(theme);
  const [addRunOpen, setAddRunOpen] = React.useState(false);
  const [runs, setRuns] = React.useState([]);

  useEffect(() => {
    const getRuns = () => {
      axios.get(`${process.env.REACT_APP_RUNS_SERVICE_URL}/runs`)
      .then((res) => {
        setRuns(res.data.data.runs);
      })
      .catch((err) => { console.log(err); });
    };

    const interval = setInterval(getRuns, 1000);

    return () => {
      clearInterval(interval);
    };
  });

  const handleClickOpen = () => {
    setAddRunOpen(true);
  };

  const handleClose = () => {
    setAddRunOpen(false);
  };

  return (
    <section className={classes.root}>
      <AddRun open={addRunOpen} onClose={handleClose}/>
      <AppBar className={classes.appBar}>
        <Toolbar>
          <Typography variant="h6">Optimization Runs</Typography>
        </Toolbar>
      </AppBar>
      <Toolbar id="back-to-top-anchor" />
      <Grid
        container
        justify="center"
        alignItems="center"
      >
        {runs.map((run) => (
          <Grid item key={run.id}>
            <Run run={run} />
          </Grid>
        ))}
      </Grid>
      <Fab
        color="secondary"
        aria-label="add"
        className={classes.fab}
        onClick={handleClickOpen}
      >
        <AddIcon/>
      </Fab>
    </section>
  );
};

const theme = createMuiTheme({
  palette: {
    type: "light",
  },
});

const ThemedApp  = () => {
  return (
    <ThemeProvider theme={theme}>
      <App/>
    </ThemeProvider>
  );
};

ReactDOM.render(
  <ThemedApp />,
  document.getElementById("root")
);