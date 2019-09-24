import React, {useEffect} from 'react';
import ReactDOM from 'react-dom';
import {
  AppBar,
  createMuiTheme,
  Fab,
  makeStyles,
  Paper,
  Toolbar,
  Typography,
  Grid, useTheme,
} from "@material-ui/core";
import { ThemeProvider } from '@material-ui/styles';
import AddIcon from "@material-ui/icons/Add";
import KeyboardArrowUpIcon from '@material-ui/icons/KeyboardArrowUp';
import axios from "axios";

import AddRun from "./components/AddRun";
import Run from "./components/Run";
import ScrollTop from "./components/ScrollTop";
import "./index.scss"

const useStyles = makeStyles(theme => ({
  root: {
    paddingBottom: theme.spacing(3),
  },
  fab: {
    position: 'absolute',
    zIndex: 1,
    top: 30,
    left: 'auto',
    right: 30,
    margin: '0 auto',
  },
  appBar: {
    top: 0,
    bottom: 'auto',
  },
  paper: {
    marginTop: -50,
    paddingTop: 50,
    paddingBottom: theme.spacing(3),
  }
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
    <section className="section">
      <AddRun open={addRunOpen} onClose={handleClose}/>
      <AppBar className={classes.appBar}>
        <Toolbar>
          <Typography variant="h6">Optimization Runs</Typography>
          <Fab
            color="secondary"
            aria-label="add"
            className={classes.fab}
            onClick={handleClickOpen}
          >
            <AddIcon/>
          </Fab>
        </Toolbar>
      </AppBar>
      <Toolbar id="back-to-top-anchor" />
      <Paper className={classes.paper}>
        <Grid
          container
          justify="center"
          alignItems="center"
        >
          {runs.map((run) => (
            <Grid item>
              <Run run={run} />
            </Grid>
          ))}
        </Grid>
      </Paper>
      <ScrollTop {...props}>
        <Fab color="secondary" size="small" aria-label="scroll back to top">
          <KeyboardArrowUpIcon />
        </Fab>
      </ScrollTop>
    </section>
  );
};

const theme = createMuiTheme({
  palette: {
    type: 'light',
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
  document.getElementById('root')
);