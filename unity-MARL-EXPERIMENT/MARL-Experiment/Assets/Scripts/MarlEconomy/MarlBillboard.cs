using UnityEngine;

namespace MarlEconomy.UnityViz
{
    // Makes an object face the main camera each frame (status orbs, labels).
    public class MarlBillboard : MonoBehaviour
    {
        private Camera cam;

        private void LateUpdate()
        {
            if (cam == null) cam = Camera.main;
            if (cam == null) return;
            transform.rotation = Quaternion.LookRotation(transform.position - cam.transform.position);
        }
    }
}
