import React from "react";
import {
  Button,
  Card,
  CardActions,
  CardActionArea,
  CardContent,
  CardMedia,
  makeStyles,
  Paper,
  Typography,
} from "@material-ui/core";

const useStyles = makeStyles(theme => ({
  card: {
    // maxWidth: 345,
    margin: theme.spacing(1),
  },
  media: {
    height: 50,
  },
}));

export default function Run(props) {
  const classes = useStyles();
  const { run } = props;

  let iteration = null;
  let f_star = null;
  let dx = null;
  let df = null;
  if (run.progress !== null) {
    iteration = run.progress["iteration"];
    const fit = run.progress["fit"];
    const pop = run.progress["pop"];
    const i_best = fit.indexOf(Math.min(...fit));
    const i_worst = fit.indexOf(Math.max(...fit));
    f_star = Math.round(1e6 * fit[i_best]) / 1e6;
    dx = Math.sqrt(
      pop[i_best].map(
        (a, i) => Math.pow(a - pop[i_worst][i], 2)
      ).reduce((a,b) => a + b, 0)
    ).toExponential(4);
    df = Math.abs(fit[i_best] - fit[i_worst]).toExponential(4);
  }


  return (
    <Card className={classes.card}>
      <CardActionArea>
        <CardMedia
          className={classes.media}
          // image="/static/images/cards/contemplative-reptile.jpg"
          title="Optimization Run"
          style={
            {
              backgroundColor: run.status === 0 ? "lightblue" : (
                run.status === 1 ? "orange" : (
                  run.status === 2 ? "lightgreen" : "red"
                )
              )
            }
          }
        >
          <Paper />
        </CardMedia>
        <CardContent>
          <Typography gutterBottom variant="body1" component="body1">
            Optimization Run
          </Typography>
          <Typography variant="body2" color="textSecondary" component="p">
            {run.status === 0 && " In Queue"}
            {run.status === 1 && " Running..."}
            {run.status === 2 && " Completed"}
            {![0, 1, 2].includes(run.status) && " Failed"}
            <br/>
            it: {iteration}/{run.gen},<br/>
            f*: {f_star},<br/>
            dx: {dx},<br/>
            df: {df}
            <br/>

          </Typography>
        </CardContent>
      </CardActionArea>
      <CardActions>
        <Button size="small" color="primary">
          Details
        </Button>
      </CardActions>
    </Card>
  );
}