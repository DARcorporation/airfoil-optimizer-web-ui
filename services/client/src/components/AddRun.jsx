import React from "react";
import axios from "axios";

import { makeStyles } from "@material-ui/core/styles";
import {
  AppBar,
  Button,
  Container,
  Dialog,
  ExpansionPanel,
  ExpansionPanelDetails,
  ExpansionPanelSummary,
  FormControl,
  FormControlLabel,
  Grid,
  IconButton,
  InputLabel,
  Select,
  MenuItem,
  Slide,
  Switch,
  TextField,
  Typography,
  Tooltip,
  Toolbar,
} from "@material-ui/core";
import CloseIcon from "@material-ui/icons/Close"
import ExpandMoreIcon from "@material-ui/icons/ExpandMore"

const Transition = React.forwardRef(function Transition(props, ref) {
  return <Slide direction="up" ref={ref} {...props} />
});

const useStyles = makeStyles(theme => ({
  dialog: {
    padding: theme.spacing(2),
    textAlign: "center",
    color: theme.palette.text.secondary,
  },
  appBar: {
    top: 0,
    bottom: "auto",
  },
  settingsContainer: {
    marginTop: 80,
    textAlign: "center",
  },
  title: {
    marginLeft: theme.spacing(2),
    flex: 1,
  },
  textField: {
    marginLeft: theme.spacing(1),
    marginRight: theme.spacing(1),
  },
  buttonGroup: {
    margin: theme.spacing(1),
  },
  heading: {
    fontSize: theme.typography.pxToRem(15),
    fontWeight: theme.typography.fontWeightRegular,
  },
  paper: {
    padding: theme.spacing(2),
    margin: theme.spacing(3),
  },
  strategyFormControl: {
    fullWidth: true,
  },
  formControl: {
    margin: theme.spacing(2),
    minWidth: 200,
  },
  tooltipWidth: {
    maxWidth: 160,
  },
}));

export default function AddRun(props) {
  const classes = useStyles();
  const { onClose, open } = props;
  const [values, setValues] = React.useState({
    cl: 1.0,
    re: 1e6,
    ma: 0.0,
    n_c: 3,
    n_t: 3,
    fix_te: true,
    gen: 1000,
    tolx: 1e-8,
    tolf: 1e-8,
    t_te_min: 0.0025,
    t_c_min: 0.1,
    r_le_min: 0.05,
    A_cs_min: "",
    Cm_max: "",
    sMutation: "rand-to-best",
    sNumber: "1",
    sCrossover: "exp",
    sRepair: "random",
    adaptivity: 2,
    f: 0.85,
    cr: 1.0,
    n_proc: 28,
    report: false,
  });
  const [TELabel, setTELabel] = React.useState("TE Thickness");
  const [batch, setBatch] = React.useState(1);
  const [expanded, setExpanded] = React.useState(false);

  const handlePanelChange = panel => (event, newExpanded) => {
    setExpanded(newExpanded ? panel : false);
  };


  const handleChange = name => event => {
    if (["fix_te", "report"].includes(name)) {
      setValues({ ...values, [name]: !values[name]});

      if (name === "fix_te") {
        if (values[name]) {
          setTELabel("Minimum TE Thickness");
        } else {
          setTELabel("TE Thickness");
        }
      }
    } else if (name === "batch") {
      setBatch(event.target.value);
    } else {
      setValues({...values, [name]: event.target.value});
    }
  };

  const handleClose = () => {
    onClose();
  };

  const addRun = event => {
    event.preventDefault();
    const data = {
      cl: values.cl,
      re: values.re,
      ma: values.ma,
      n_c: values.n_c,
      n_t: values.n_t,
      fix_te: values.fix_te,
      gen: values.gen,
      tolx: values.tolx,
      tolf: values.tolf,
      t_te_min: values.t_te_min ? Number(values.t_te_min) : null,
      t_c_min: values.t_c_min ? Number(values.t_c_min) : null,
      r_le_min: values.r_le_min ? Number(values.r_le_min) : null,
      A_cs_min: values.A_cs_min ? Number(values.A_cs_min) : null,
      Cm_max: values.Cm_max ? Number(values.Cm_max) : null,
      adaptivity: values.adaptivity,
      f: values.adaptivity === 0 ? values.f : null,
      cr: values.adaptivity === 0 ? values.cr : null,
      strategy: [values.sMutation, values.sNumber, values.sCrossover, values.sRepair].join("/"),
      n_proc: values.n_proc,
      report: values.report,
    };
    for (let i = 0; i < batch; i++) {
      axios.post(`${process.env.REACT_APP_RUNS_SERVICE_URL}/runs`, data)
        .then((res) => {})
        .catch((err) => { console.log(err); });
    }
    onClose();
  };

  return (

    <Dialog
      open={open}
      onClose={handleClose}
      scroll="paper"
      fullScreen={true}
      TransitionComponent={Transition}
      keepMounted
    >
      <AppBar className={classes.appBar}>
        <Toolbar>
          <IconButton edge="start" color="inherit" onClick={handleClose}>
            <CloseIcon />
          </IconButton>
          <Typography variant="h6" className={classes.title}>
            New Run
          </Typography>
          <Button form="add-run-form" color="inherit" type="submit">
            Submit
          </Button>
        </Toolbar>
      </AppBar>
      <Container className={classes.settingsContainer}>
        <form onSubmit={(event) => addRun(event)} id="add-run-form">
          <ExpansionPanel expanded={expanded === "panel1"} onChange={handlePanelChange("panel1")}>
            <ExpansionPanelSummary
              expandIcon={<ExpandMoreIcon />}
            >
              <Typography variant="subtitle1">Basic Problem Setup</Typography>
            </ExpansionPanelSummary>
            <ExpansionPanelDetails>
              <Container>
                <TextField
                  label="Design Lift Coefficient"
                  name="cl"
                  value={values.cl}
                  onChange={handleChange("cl")}
                  type="number"
                  inputProps={{
                    step: 0.001,
                  }}
                  className={classes.textField}
                  InputLabelProps={{
                    shrink: true,
                  }}
                  margin="normal"
                  variant="outlined"
                />
                <br/>
                <TextField
                  label="Reynolds Number"
                  name="re"
                  value={values.re}
                  onChange={handleChange("re")}
                  type="number"
                  inputProps={{
                    step: 0.001,
                  }}
                  className={classes.textField}
                  InputLabelProps={{
                    shrink: true,
                  }}
                  margin="normal"
                  variant="outlined"
                />
                <br/>
                <TextField
                  label="Mach Number"
                  name="ma"
                  value={values.ma}
                  onChange={handleChange("ma")}
                  type="number"
                  inputProps={{
                    step: 0.001,
                  }}
                  className={classes.textField}
                  InputLabelProps={{
                    shrink: true,
                  }}
                  margin="normal"
                  variant="outlined"
                />
                <br/>
                <FormControlLabel control={
                  <Switch
                    checked={values.fix_te}
                    onChange={handleChange("fix_te")}
                    value="fix_te"
                    name="fix_te"
                    inputProps={{ "aria-label": "primary checkbox" }}
                  />} label="Fix TE Thickness?" //labelPlacement="top"
                />
                <br/>
                <TextField
                  label={TELabel}
                  name="t_te_min"
                  value={values.t_te_min}
                  onChange={handleChange("t_te_min")}
                  type="text"
                  inputProps={{
                    pattern: "-?([1-9]\\d*|0)+(\\.\\d*)?([Ee][+-]?\\d+)?",
                  }}
                  className={classes.textField}
                  InputLabelProps={{
                    shrink: true,
                  }}
                  margin="normal"
                  variant="outlined"
                />
                <br/>
                <TextField
                  label="Minimum t/c"
                  name="t_c_min"
                  value={values.t_c_min}
                  onChange={handleChange("t_c_min")}
                  type="text"
                  inputProps={{
                    pattern: "[+-]?([1-9]\\d*|0)?(\\.\\d*)?([Ee][+-]?\\d+)?",
                  }}
                  className={classes.textField}
                  InputLabelProps={{
                    shrink: true,
                  }}
                  margin="normal"
                  variant="outlined"
                />
                <br/>
                <TextField
                  label="Minimum LE Radius"
                  name="r_le_min"
                  value={values.r_le_min}
                  onChange={handleChange("r_le_min")}
                  type="text"
                  inputProps={{
                    pattern: "[+-]?([1-9]\\d*|0)?(\\.\\d*)?([Ee][+-]?\\d+)?",
                  }}
                  className={classes.textField}
                  InputLabelProps={{
                    shrink: true,
                  }}
                  margin="normal"
                  variant="outlined"
                />
                <br/>
                <TextField
                  label="Minimum Area"
                  name="A_cs_min"
                  value={values.A_cs_min}
                  onChange={handleChange("A_cs_min")}
                  type="text"
                  inputProps={{
                    pattern: "[+-]?([1-9]\\d*|0)?(\\.\\d*)?([Ee][+-]?\\d+)?",
                  }}
                  className={classes.textField}
                  InputLabelProps={{
                    shrink: true,
                  }}
                  margin="normal"
                  variant="outlined"
                />
                <br/>
                <TextField
                  label="Maximum Absolute Cm"
                  name="Cm_max"
                  value={values.Cm_max}
                  onChange={handleChange("Cm_max")}
                  type="text"
                  inputProps={{
                    pattern: "[+-]?([1-9]\\d*|0)?(\\.\\d*)?([Ee][+-]?\\d+)?",
                  }}
                  className={classes.textField}
                  InputLabelProps={{
                    shrink: true,
                  }}
                  margin="normal"
                  variant="outlined"
                />
              </Container>
            </ExpansionPanelDetails>
          </ExpansionPanel>
          <ExpansionPanel expanded={expanded === "panel2"} onChange={handlePanelChange("panel2")}>
            <ExpansionPanelSummary
              expandIcon={<ExpandMoreIcon />}
            >
              <Typography variant="subtitle1">Number of Design Variables</Typography>
            </ExpansionPanelSummary>
            <ExpansionPanelDetails>
              <Container>
                <TextField
                  label="Mean Chord Line"
                  name="n_c"
                  value={values.n_c}
                  onChange={handleChange("n_c")}
                  type="number"
                  inputProps={{
                    min: 1,
                  }}
                  className={classes.textField}
                  InputLabelProps={{
                    shrink: true,
                  }}
                  margin="normal"
                  variant="outlined"
                />
                <br/>
                <TextField
                  label="Thickness Distribution"
                  name="n_t"
                  value={values.n_t}
                  onChange={handleChange("n_t")}
                  type="number"
                  inputProps={{
                    min: 1,
                  }}
                  className={classes.textField}
                  InputLabelProps={{
                    shrink: true,
                  }}
                  margin="normal"
                  variant="outlined"
                />
              </Container>
            </ExpansionPanelDetails>
          </ExpansionPanel>
          <ExpansionPanel expanded={expanded === "panel3"} onChange={handlePanelChange("panel3")}>
            <ExpansionPanelSummary
              expandIcon={<ExpandMoreIcon />}
            >
              <Typography variant="subtitle1">Termination Settings</Typography>
            </ExpansionPanelSummary>
            <ExpansionPanelDetails>
              <Container>
                <TextField
                  label="Number of Generations"
                  name="gen"
                  value={values.gen}
                  onChange={handleChange("gen")}
                  type="number"
                  inputProps={{
                    min: 0,
                  }}
                  className={classes.textField}
                  InputLabelProps={{
                    shrink: true,
                  }}
                  margin="normal"
                  variant="outlined"
                />
                <br/>
                <TextField
                  label="Design Vector Tolerance"
                  name="tolx"
                  value={values.tolx}
                  onChange={handleChange("tolx")}
                  type="text"
                  inputProps={{
                    pattern: "[+-]?([1-9]\\d*|0)?(\\.\\d*)?([Ee][+-]?\\d+)?",
                  }}
                  className={classes.textField}
                  InputLabelProps={{
                    shrink: true,
                  }}
                  margin="normal"
                  variant="outlined"
                />
                <br/>
                <TextField
                  label="Objective Function Tolerance"
                  name="tolf"
                  value={values.tolf}
                  onChange={handleChange("tolf")}
                  type="text"
                  inputProps={{
                    pattern: "[+-]?([1-9]\\d*|0)?(\\.\\d*)?([Ee][+-]?\\d+)?",
                  }}
                  className={classes.textField}
                  InputLabelProps={{
                    shrink: true,
                  }}
                  margin="normal"
                  variant="outlined"
                />
              </Container>
            </ExpansionPanelDetails>
          </ExpansionPanel>
          <ExpansionPanel expanded={expanded === "panel4"} onChange={handlePanelChange("panel4")}>
            <ExpansionPanelSummary
              expandIcon={<ExpandMoreIcon />}
            >
              <Typography variant="subtitle1">Advanced Controls</Typography>
            </ExpansionPanelSummary>
            <ExpansionPanelDetails>
              <Container maxWidth="sm">
                <Typography variant="body2" component="body2">Evolution Strategy</Typography>
                <br/>
                <Grid container>
                  <Grid key="mutation" item xs={4}>
                    <FormControl variant="outlined" fullWidth={true}>
                      <Select
                        value={values.sMutation}
                        onChange={handleChange("sMutation")}
                      >
                        <MenuItem value="rand">rand</MenuItem>
                        <MenuItem value="best">best</MenuItem>
                        <MenuItem value="rand-to-best">rand-to-best</MenuItem>
                      </Select>
                    </FormControl>
                  </Grid>
                  <Grid key="number" item xs={2}>
                    <FormControl variant="outlined" fullWidth={true}>
                      <Select
                        value={values.sNumber}
                        onChange={handleChange("sNumber")}
                      >
                        <MenuItem value="1">1</MenuItem>
                        <MenuItem value="2">2</MenuItem>
                        <MenuItem value="3">3</MenuItem>
                      </Select>
                    </FormControl>
                  </Grid>
                  <Grid key="crossover" item xs={3}>
                    <FormControl variant="outlined" fullWidth={true}>
                      <Select
                        value={values.sCrossover}
                        onChange={handleChange("sCrossover")}
                      >
                        <MenuItem value="bin">bin</MenuItem>
                        <MenuItem value="exp">exp</MenuItem>
                      </Select>
                    </FormControl>
                  </Grid>
                  <Grid key="repair" item xs={3}>
                    <FormControl variant="outlined" fullWidth={true}>
                      <Select
                        value={values.sRepair}
                        onChange={handleChange("sRepair")}
                      >
                        <MenuItem value="random">random</MenuItem>
                        <MenuItem value="clip">clip</MenuItem>
                      </Select>
                    </FormControl>
                  </Grid>
                </Grid>
                <br/>
                <FormControl variant="outlined" className={classes.formControl}>
                  <InputLabel>Self-Adaptivity</InputLabel>
                  <Select
                    value={values.adaptivity}
                    onChange={handleChange("adaptivity")}
                  >
                    <Tooltip
                      title="The Mutation and Crossover Rates are kept fixed at the specified values."
                      value={0}
                      placement="left"
                      classes={{ tooltip: classes.tooltipWidth }}
                    >
                      <MenuItem value={0}>No Adaptivity</MenuItem>
                    </Tooltip>
                    <Tooltip
                      title="Simple Adaptivity uses a Monte-Carlo search to find the optimal Mutation and Crossover
                      Rates during the course of the optimization."
                      value={1}
                      placement="left"
                      classes={{ tooltip: classes.tooltipWidth }}
                    >
                      <MenuItem value={1}>Simple Adaptivity</MenuItem>
                    </Tooltip>
                    <Tooltip
                      title="Complex Adaptivity mutates the Mutation and Crossover Rates with the same mutation
                      strategy as the population."
                      value={2}
                      placement="left"
                      classes={{ tooltip: classes.tooltipWidth }}
                    >
                      <MenuItem value={2}>Complex Adaptvity</MenuItem>
                    </Tooltip>
                  </Select>
                </FormControl>
                <br/>
                {values.adaptivity === 0 &&
                  <React.Fragment>
                    <TextField
                      label="Mutation Rate"
                      name="f"
                      value={values.f}
                      onChange={handleChange("f")}
                      type="number"
                      inputProps={{
                        min: 0,
                        max: 1,
                        step: 0.001,
                      }}
                      className={classes.textField}
                      InputLabelProps={{
                        shrink: true,
                      }}
                      margin="normal"
                      variant="outlined"
                    />
                    <br/>
                    <TextField
                      label="Crossover Rate"
                      name="cr"
                      value={values.cr}
                      onChange={handleChange("cr")}
                      type="number"
                      inputProps={{
                        min: 0,
                        max: 1,
                        step: 0.001,
                      }}
                      className={classes.textField}
                      InputLabelProps={{
                        shrink: true,
                      }}
                      margin="normal"
                      variant="outlined"
                    />
                  </React.Fragment>
                }
              </Container>
            </ExpansionPanelDetails>
          </ExpansionPanel>
          <ExpansionPanel expanded={expanded === "panel5"} onChange={handlePanelChange("panel5")}>
            <ExpansionPanelSummary
              expandIcon={<ExpandMoreIcon />}
            >
              <Typography variant="subtitle1">Miscellaneous</Typography>
            </ExpansionPanelSummary>
            <ExpansionPanelDetails>
              <Container>
                <TextField
                  label="Number of Processors"
                  name="n_proc"
                  value={values.n_proc}
                  onChange={handleChange("n_proc")}
                  type="number"
                  inputProps={{
                    min: 1,
                  }}
                  className={classes.textField}
                  InputLabelProps={{
                    shrink: true,
                  }}
                  margin="normal"
                  variant="outlined"
                />
                <br/>
                <TextField
                  label="Batch Submit"
                  name="batch"
                  value={batch}
                  onChange={handleChange("batch")}
                  type="number"
                  inputProps={{
                    min: 1,
                  }}
                  className={classes.textField}
                  InputLabelProps={{
                    shrink: true,
                  }}
                  margin="normal"
                  variant="outlined"
                />
                <br/>
                <FormControlLabel control={
                  <Switch
                    checked={values.report}
                    onChange={handleChange("report")}
                    value="report"
                    name="report"
                    inputProps={{ "aria-label": "primary checkbox" }}
                  />} label="Report Results via Email?" //labelPlacement="top"
                />
              </Container>
            </ExpansionPanelDetails>
          </ExpansionPanel>
        </form>
      </Container>
    </Dialog>
  );
}