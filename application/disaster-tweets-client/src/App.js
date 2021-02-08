import React from "react";
import { BrowserRouter as Router, Switch, Route, Link } from "react-router-dom";

import EDA from "./pages/EDA.js";
import Prediction from "./pages/Prediction.js";

import Container from "@material-ui/core/Container";
import CssBaseline from "@material-ui/core/CssBaseline";
import { makeStyles } from "@material-ui/core/styles";

const useStyles = makeStyles((theme) => ({
  paper: {
    marginTop: theme.spacing(8),
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
  },
  link: {
    margin: theme.spacing(1),
    textDecoration: "none",
    fontSize: 20,
  },
}));

const App = () => {
  const classes = useStyles();

  return (
    <Container>
      <CssBaseline />
      <div className={classes.paper}>
        <Router>
          <div>
            <Link className={classes.link} to="/prediction">
              Prediction
            </Link>
            <Link className={classes.link} to="/exploratory-data-analysis">
              Exploratory data analysis
            </Link>
          </div>

          <div>
            <Switch>
              <Route exact path="/prediction" component={Prediction}></Route>
              <Route path="/exploratory-data-analysis" component={EDA}></Route>
            </Switch>
          </div>
        </Router>
      </div>
    </Container>
  );
};

export default App;
