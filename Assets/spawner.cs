using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class spawner : MonoBehaviour {
	public GameObject[] balls;    

	public void Start () {
		// Start calling the Spawn function repeatedly after a delay .
		// InvokeRepeating("Spawn", Time.deltaTime * 0.2);
	}

	void Spawn () {
		// Instantiate a random enemy.
		int ballIndex = Random.Range(0, balls.Length);
		Instantiate(balls[ballIndex], transform.position, transform.rotation);
	}
}