1. Need to understand when reauctioning needs to happen and when replanning needs to happen and 
    rewrite the code (SSIA, SSCIA, collateral)
  - SSIA_Main does currently. Why do the other s not do?
  - ` Reauction trigger is still overstated`: In the current SSIA-family code, 
      ordinary obstacle discovery does not trigger a full global reauction.
  - We should not reauction when there is a new obstacle, only replan. set mode to replan, if replan fails then re-add task to queue

2. For the Issue with the different reward values, let that be for now
3. 
